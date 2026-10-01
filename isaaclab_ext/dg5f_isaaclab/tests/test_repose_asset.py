"""Prove the benchmark adapter preserves geometry, masses and all movable axes."""
import importlib.util
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / 'source/dg5f_isaaclab/dg5f_isaaclab/tasks/direct/dg5f_repose/asset_adapter.py'
spec = importlib.util.spec_from_file_location('repose_adapter', MODULE)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)
SOURCE = ROOT.parents[1] / 'models/dg5f/urdf/dg5f_right.urdf'


def test_fixed_disabled_axis_preserves_every_other_joint_and_all_links(tmp_path):
    destination = tmp_path / 'hand.urdf'
    adapter.fixed_axis_urdf(SOURCE, destination, 'rj_dg_5_1')
    before, after = ET.parse(SOURCE).getroot(), ET.parse(destination).getroot()
    for mesh in before.iter('mesh'):
        mesh.set('filename', str((SOURCE.parent / mesh.attrib['filename']).resolve()))
    originals = {j.attrib['name']: j for j in before.findall('joint')}
    changed = {j.attrib['name']: j for j in after.findall('joint')}
    assert originals.keys() == changed.keys()
    for name in originals:
        if name != 'rj_dg_5_1':
            assert ET.tostring(originals[name]) == ET.tostring(changed[name])
    locked = changed['rj_dg_5_1']
    assert locked.get('type') == 'fixed'
    for tag in ('parent', 'child', 'origin'):
        assert ET.tostring(originals['rj_dg_5_1'].find(tag)) == ET.tostring(locked.find(tag))
    assert [ET.tostring(link) for link in before.findall('link')] == [ET.tostring(link) for link in after.findall('link')]
    assert sum(j.get('type') == 'revolute' for j in after.findall('joint')) == 19
    for mesh in after.iter('mesh'):
        assert Path(mesh.get('filename')).is_file()
    mtime = destination.stat().st_mtime_ns
    adapter.fixed_axis_urdf(SOURCE, destination, 'rj_dg_5_1')
    assert destination.stat().st_mtime_ns == mtime  # do not spuriously trigger expensive USD conversion
