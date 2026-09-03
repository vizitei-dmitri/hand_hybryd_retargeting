# tesollo_control
Wrapper class for easy control of tesollo gripper 5f

## System

- Ubuntu 22
- Tesollo Delto 5f right

## Requirements

### Eigen3

```bash
sudo apt install libeigen3-dev
```

## Python installation

The Python bindings can be built and installed directly from the repository:

```bash
sudo apt install build-essential cmake libeigen3-dev
python3 -m pip install .
```

For an editable development installation, use:

```bash
python3 -m pip install -e .
```

The package installs the `dg5f_python` extension together with the required
Tesollo SDK shared library. It currently supports 64-bit Linux, matching the
bundled vendor library.

```python
import dg5f_python

gripper = dg5f_python.DGApi.instance()
```

## 60 Hz sine position test

Install the plotting dependency and run the example from the repository:

```bash
python3 -m pip install '.[examples]'
python3 examples/sine_delta_test.py
```

Configure the test by editing the constants at the top of the script:

```python
JOINT = 2
AMPLITUDE_DEG = 5.0
SINE_FREQUENCY_HZ = 0.5
COMMAND_RATE_HZ = 60.0
ZERO_TIME_S = 2.0
TEST_DURATION_S = 10.0
```

The gripper is first commanded to the all-zero position, then the selected
joint receives `amplitude * sin(2*pi*frequency*time)`. The test returns the
gripper to zero even after `Ctrl+C`, and writes `sine_delta_test.csv` plus
`sine_delta_test.png`. The graph contains commanded/measured position,
tracking error, command delta per sample, and the actual Python loop period.

Start with a small amplitude and use a joint whose mechanical limits include
the full interval `[-amplitude, +amplitude]`. At 60 Hz, the theoretical peak
command delta between samples is:

```text
2 * amplitude * sin(pi * frequency / 60)
```

The gripper IP, port, slave ID, and output filename are configured in the same
block.
