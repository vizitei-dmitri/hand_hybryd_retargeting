import json
from dataclasses import asdict
import subprocess
import sys
import time

import numpy as np
import pytest
import torch

from kinect_mano.fitting import FitConfig, ManoFitter
from kinect_mano.io import load_observations, save_observations, save_reconstruction, state_mapping
from kinect_mano.model import ManoModel
from kinect_mano.pipeline import LatestPairSource, fit_sequence
from kinect_mano.synthetic import generate_sequence, make_fixture_model

torch.set_num_threads(1)


def test_observations_preserve_per_camera_geometry_and_truth(tmp_path):
    model = ManoModel(make_fixture_model(), allow_test_model=True)
    observations, truth = generate_sequence(model, frames=2, points_per_camera=8)
    path = tmp_path/"observations.npz"
    save_observations(path,observations,truth=truth,metadata={"model_type":"numerical_fixture"})
    restored, restored_truth, metadata = load_observations(path)
    assert metadata["model_type"] == "numerical_fixture"
    for expected, actual in zip(observations,restored):
        np.testing.assert_allclose(actual.joints_m,expected.joints_m)
        for a,b in zip(actual.views,expected.views):
            assert a.camera_id == b.camera_id
            np.testing.assert_allclose(a.points_m,b.points_m,atol=1e-7)
            np.testing.assert_allclose(a.camera_center_m,b.camera_center_m)
    np.testing.assert_equal(restored_truth["joints_m"],truth["joints_m"])
    with pytest.raises(FileExistsError):
        save_observations(path, observations)


def test_checkpoint_resume_and_export_parameters_recreate_geometry(tmp_path):
    model = ManoModel(make_fixture_model(),allow_test_model=True)
    observations, truth = generate_sequence(model,frames=3,points_per_camera=0)
    config = FitConfig(mode="joint",iterations=8)
    fingerprint = {"input":"synthetic-fixture", "config":asdict(config)}
    initial = fit_sequence(ManoFitter(model,config),observations[:1],tmp_path,fingerprint=fingerprint)
    # Simulate an interrupted last line, never a complete frame.
    with (tmp_path/"checkpoint.jsonl").open("a") as stream:
        stream.write('{"partial":')
    states = fit_sequence(ManoFitter(model,config),observations,tmp_path,fingerprint=fingerprint,resume=True)
    assert len(states) == 3
    np.testing.assert_equal(states[0].theta,initial[0].theta)
    with pytest.raises(ValueError,match="mismatch"):
        fit_sequence(ManoFitter(model,config),observations,tmp_path,fingerprint={"wrong":True},resume=True)
    path = save_reconstruction(tmp_path,states,model,config,observations=observations)
    with np.load(path,allow_pickle=False) as result:
        assert result["valid"].shape == (3,)
        with torch.no_grad():
            vertices,joints = model(*(torch.tensor(result[k],dtype=model.dtype) for k in ("theta","beta","global_orient","translation_m")))
        np.testing.assert_allclose(result["vertices_m"],vertices.numpy(),atol=1e-7)
        np.testing.assert_allclose(result["joints_m"],joints.numpy(),atol=1e-7)
        assert json.loads(str(result["metadata_json"]))["model_type"] == "numerical_fixture"


def test_state_mapping_is_json_and_preserves_separate_clouds():
    model=ManoModel(make_fixture_model(),allow_test_model=True)
    observations,_=generate_sequence(model,frames=1,points_per_camera=4)
    state=ManoFitter(model,FitConfig(mode="joint",iterations=2)).fit(observations[0])
    mapping=state_mapping(state,model.faces.numpy(),observation=observations[0],metadata=model.metadata)
    assert set(mapping["clouds_m"]) == {v.camera_id for v in observations[0].views}
    assert len(mapping["cloud_m"]) == 8
    assert mapping["metadata"]["model_type"] == "numerical_fixture"
    json.dumps(mapping,allow_nan=False)


def test_capture_thread_overwrites_queue_and_propagates_errors():
    class Capture:
        count=0
        def read_pair(self,**kwargs):
            time.sleep(.002)
            self.count+=1
            return self.count,self.count
    capture=Capture()
    source=LatestPairSource(capture,tolerance_s=.005).start()
    try:
        first,_,_=source.next()
        time.sleep(.025)
        second,pair,_=source.next(first)
        assert second > first+1
        assert pair == (second,second)
        assert not hasattr(source,"queue")
    finally:
        source.close()
    assert not source.thread.is_alive()
    class BadCapture:
        def read_pair(self,**kwargs):
            raise RuntimeError("sensor unplugged")
    bad=LatestPairSource(BadCapture(),tolerance_s=.005).start()
    try:
        with pytest.raises(RuntimeError,match="sensor unplugged"):
            bad.next()
    finally:
        bad.close()


def test_cli_missing_model_fails_without_synthetic_fallback(tmp_path):
    result=subprocess.run([sys.executable,"-m","kinect_mano","fit","--model",str(tmp_path/"missing.npz"),"--input","also-missing.npz","--output",str(tmp_path/"out")],capture_output=True,text=True)
    assert result.returncode == 2
    assert "MANO model missing" in result.stderr
    assert not (tmp_path/"out").exists()


def test_cli_synthetic_fit_and_headless_view(tmp_path):
    def run(*args):
        process=subprocess.run([sys.executable,"-m","kinect_mano",*map(str,args)],capture_output=True,text=True,timeout=30)
        assert process.returncode == 0,process.stderr
        return json.loads(process.stdout)
    run("demo","--output",tmp_path/"input","--frames",1,"--points-per-camera",8)
    result=run("fit","--model",tmp_path/"input/numerical_fixture.npz","--allow-test-model","--device","cpu","--input",tmp_path/"input/observations.npz","--output",tmp_path/"fit","--iterations",4)
    assert result["model_type"] == "numerical_fixture"
    assert (tmp_path/"fit/preview.png").is_file()
    run("view","--input",tmp_path/"fit/reconstruction.npz","--output",tmp_path/"frame.png")
    assert (tmp_path/"frame.png").stat().st_size > 1000
