#!/usr/bin/env python3
"""Host-only passive network capture plus Docker ROS recorder orchestration.

No SDK client. No network reconfiguration. Only ping, tcpdump and sysfs reads.
"""

import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tarfile
import time


class NetworkCapture:
    def __init__(self, directory, interface, ip, ping=False, tcpdump=False):
        self.directory = Path(directory)
        self.interface = interface
        self.started = time.monotonic()
        self.previous_link = None
        self.processes = []
        self.status = {"ping_enabled": False, "tcpdump_enabled": False,
                       "start_wall_time_unix_s": time.time(),
                       "start_monotonic_s": self.started}
        self.log = (self.directory / "host_network.jsonl").open("w")
        if ping:
            self.start_process("ping", ["ping", "-D", "-i", "0.1", ip], "ping.log")
        if tcpdump:
            command = ["tcpdump", "-i", interface, "-nn", "-U", "-w",
                       str(self.directory / "dg5f_tcp.pcap"),
                       "host", ip, "and", "tcp", "port", "502"]
            privileged = os.geteuid() != 0
            if privileged:
                command = ["sudo", "-n", *command]
            self.start_process("tcpdump", command, "tcpdump.log", privileged)

    def start_process(self, name, command, filename, privileged=False):
        output = (self.directory / filename).open("w")
        if shutil.which(name) is None:
            output.write(f"{name} unavailable: executable not found\n")
            output.close()
            self.status[f"{name}_error"] = "executable not found"
            print(f"{name} unavailable; other logs continue", flush=True)
            return
        try:
            process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            self.processes.append((name, process, output, privileged))
        except OSError as error:
            output.write(str(error))
            output.close()
            self.status[f"{name}_error"] = str(error)

    def sample(self):
        row = {"time_s": time.monotonic() - self.started,
               "monotonic_s": time.monotonic(), "wall_time_unix_s": time.time()}
        base = Path("/sys/class/net") / self.interface
        for key in ("rx_packets", "tx_packets", "rx_errors", "tx_errors",
                    "rx_dropped", "tx_dropped", "carrier", "operstate", "carrier_changes"):
            path = base / key if key in {"carrier", "operstate", "carrier_changes"} else base / "statistics" / key
            try:
                raw = path.read_text().strip()
                row[key] = raw if key == "operstate" else int(raw)
            except (OSError, ValueError):
                row[key] = None
        link = (row["carrier"], row["operstate"])
        if self.previous_link is not None and link != self.previous_link:
            row["event"] = "LINK_CHANGED"
        self.previous_link = link
        self.log.write(json.dumps(row) + "\n")
        self.log.flush()
        for name, process, _, _ in self.processes:
            running = process.poll() is None
            self.status[f"{name}_enabled"] = running
            if not running:
                self.status[f"{name}_exit_code"] = process.returncode
                if not self.status.get(f"{name}_reported_unavailable"):
                    print(f"{name} capture unavailable; see {name}.log. Other logs continue.", flush=True)
                    self.status[f"{name}_reported_unavailable"] = True
        (self.directory / "network_status.json").write_text(json.dumps(self.status, indent=2))

    def stop(self):
        self.sample()
        for name, process, output, privileged in self.processes:
            if process.poll() is None:
                if privileged:
                    # Signal only the sudo process created above; sudo forwards it.
                    subprocess.run(["sudo", "-n", "kill", "-INT", str(process.pid)],
                                   check=False, timeout=3)
                else:
                    process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    if privileged:
                        subprocess.run(["sudo", "-n", "kill", "-TERM", "--", f"-{process.pid}"],
                                       check=False, timeout=3)
                    else:
                        process.terminate()
                    process.wait(timeout=3)
            self.status[f"{name}_exit_code"] = process.returncode
            output.close()
        self.log.close()
        (self.directory / "network_status.json").write_text(json.dumps(self.status, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ping", action="store_true")
    parser.add_argument("--tcpdump", action="store_true")
    parser.add_argument("--interface", default="enp49s0")
    parser.add_argument("--hand-ip", default="169.254.186.72")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{os.getpid()}"
    directory = project / "debug_runs" / stamp
    flags = [flag for flag, enabled in (("--ping", args.ping), ("--tcpdump", args.tcpdump)) if enabled]
    stop_path = f"/workspace/debug_runs/{stamp}/.stop_recording"
    command = ["docker", "compose", "-f", str(project / "compose.yaml"), "exec", "-T",
               "lerobot_hand", "bash", "-lc",
               'source /opt/ros/humble/setup.bash; source /workspace/install/setup.bash; exec python3 -m lerobot_robot_dg5f.debug_recorder "$@"',
               "recorder", "--run-stamp", stamp, "--external-network", "--defer-archive",
               "--stop-file", stop_path, "--interface", args.interface, "--hand-ip", args.hand_ip, *flags]
    process = subprocess.Popen(command, start_new_session=True)
    network = None
    interrupted = False

    def request_stop(*_):
        nonlocal interrupted
        interrupted = True
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        deadline = time.monotonic() + 30
        while not (directory / "manifest.json").exists():
            if process.poll() is not None or time.monotonic() > deadline or interrupted:
                raise RuntimeError("ROS recorder did not initialize")
            time.sleep(0.1)
        network = NetworkCapture(directory, args.interface, args.hand_ip, args.ping, args.tcpdump)
        print(f"Host network capture: {directory}\nPress Ctrl+C to finalize one archive.", flush=True)
        while not interrupted and process.poll() is None:
            network.sample()
            time.sleep(0.5)
    finally:
        if network:
            network.stop()
        if directory.exists():
            (directory / ".stop_recording").touch()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            print("ROS recorder did not finalize. Logs preserved; no complete archive claimed.")
            raise
        if not (directory / "summary.txt").exists():
            raise RuntimeError(f"No finalized summary; raw files remain in {directory}")
        if network:
            manifest_path = directory / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["host_network_capture"] = network.status
            manifest_path.write_text(json.dumps(manifest, indent=2))
            with (directory / "summary.txt").open("a") as output:
                output.write("\nhost_network_capture: " + json.dumps(network.status) + "\n")
        archive = directory.parent / f"dg5f_debug_{stamp}.tar.gz"
        with tarfile.open(archive, "w:gz") as bundle:
            bundle.add(directory, arcname=stamp)
        print(f"Archive: {archive}", flush=True)


if __name__ == "__main__":
    main()
