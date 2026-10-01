"""CPU-only generation of the isolated, already-disabled DG5F axis representation."""
from pathlib import Path
import xml.etree.ElementTree as ET


def fixed_axis_urdf(source: Path, destination: Path, disabled: str):
    """Generate an isolated importer input; never edit the real-robot model."""
    model = ET.parse(source)
    matches = [joint for joint in model.getroot().findall('joint') if joint.get('name') == disabled]
    if len(matches) != 1:
        raise ValueError(f'Expected one disabled joint {disabled}')
    joint = matches[0]
    joint.set('type', 'fixed')
    for child in list(joint):
        if child.tag in ('axis', 'limit', 'dynamics'):
            joint.remove(child)
    for mesh in model.getroot().iter('mesh'):
        mesh.set('filename', str((source.parent / mesh.attrib['filename']).resolve()))
    data = ET.tostring(model.getroot(), encoding='unicode') + '\n'
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists() or destination.read_text() != data:
        destination.write_text(data)
    return str(destination)
