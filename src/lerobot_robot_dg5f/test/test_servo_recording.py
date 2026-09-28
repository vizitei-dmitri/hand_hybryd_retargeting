import csv
import json
from types import SimpleNamespace

from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from std_msgs.msg import String

from lerobot_robot_dg5f.debug_recording import DebugRunWriter
from lerobot_robot_dg5f.debug_recorder import Dg5fDebugRecorder, _parse_diagnostics


def test_each_servo_tick_and_tracking_cause_survive_bundle(tmp_path):
    writer=DebugRunWriter(tmp_path,{},run_stamp='servo')
    recorder=SimpleNamespace(writer=writer)
    for index in range(60):
        Dg5fDebugRecorder._on_servo(recorder,String(data=json.dumps(dict(
            source_monotonic_s=index/60, target=[70]*20, previous_q_cmd=[-50]*20,
            raw_servo_step=[2]*20,new_q_cmd=[-48]*20,command_direction_reason=['NONE']*20))))
    writer.record(dict(last_landmarks_age_ms=500,last_retarget_age_ms=500,
                       tracking_loss_reason='STALE_LANDMARKS',tracking_loss_detail='QUEST_INPUT_STALE'))
    writer.finalize()
    ticks=[json.loads(row) for row in (writer.run_dir/'servo_ticks.jsonl').read_text().splitlines()]
    assert len(ticks)==60
    assert ticks[0]['raw_servo_step']==[2]*20
    with (writer.run_dir/'timeline.csv').open() as file:
        row=next(csv.DictReader(file))
    assert row['tracking_loss_reason']=='STALE_LANDMARKS'
    assert row['tracking_loss_detail']=='QUEST_INPUT_STALE'


def test_contact_state_arrays_are_not_silently_dropped_from_diagnostics():
    values=[KeyValue(key='object_contact_state',value='FREE,CONTACT_HOLD,FREE,FREE,FREE'),
            KeyValue(key='object_contact_active',value='False,True,False,False,False'),
            KeyValue(key='contact_anchor_measured',value=','.join(['12.0']*20))]
    message=DiagnosticArray(status=[DiagnosticStatus(name='dg5f_lerobot_bridge',values=values)])
    decoded=_parse_diagnostics(message)
    assert decoded['object_contact_state'][1]=='CONTACT_HOLD'
    assert decoded['object_contact_active']==[False,True,False,False,False]
    assert decoded['contact_anchor_measured']==[12.0]*20


def test_runtime_events_and_physical_lead_survive_recorder(tmp_path):
    writer = DebugRunWriter(tmp_path, {}, run_stamp='policy')
    recorder = SimpleNamespace(writer=writer)
    names = ['TRACKING_LOST', 'TRACKING_RECOVERED', 'CURRENT_LOAD_THRESHOLD',
             'SDK_ERROR', 'CONTACT_PENDING', 'CONTACT_LATCHED', 'CONTACT_RELEASED',
             'COMMAND_DIRECTION_VIOLATION_BLOCKED']
    for name in names:
        Dg5fDebugRecorder._on_event(recorder, String(data=json.dumps(dict(event=name, last_motion_result=500))))
    values = [KeyValue(key='actual_command_lead_deg', value=','.join(['1.0']*20)),
              KeyValue(key='runtime_state', value='TRACKING_HOLD')]
    decoded = _parse_diagnostics(DiagnosticArray(status=[DiagnosticStatus(name='dg5f_lerobot_bridge', values=values)]))
    writer.record(decoded)
    writer.finalize()
    events = [json.loads(line) for line in (writer.run_dir/'events.jsonl').read_text().splitlines()]
    assert all(any(event['event'] == name for event in events) for name in names)
    with (writer.run_dir/'timeline.csv').open() as stream:
        row = next(csv.DictReader(stream))
    assert row['runtime_state'] == 'TRACKING_HOLD'
    assert row['actual_command_lead_deg_6'] == '1.0'


def test_teleop_capture_keeps_original_stamp_points_and_ros_logs(tmp_path):
    from geometry_msgs.msg import Point, Pose, PoseArray
    from vr_haptic_msgs.msg import ManoLandmarks
    from rcl_interfaces.msg import Log
    writer=DebugRunWriter(tmp_path,{},run_stamp='teleop')
    writer.enable_teleop_recording()
    node=SimpleNamespace(writer=writer)
    raw=ManoLandmarks()
    raw.header.stamp.sec=123
    raw.header.stamp.nanosec=456
    raw.landmarks=[Point(x=1.,y=2.,z=3.)]*20  # Keep invalid frames for diagnosis too.
    normalized=PoseArray(header=raw.header,poses=[Pose(position=Point(x=4.))]*21)
    Dg5fDebugRecorder._on_teleop_input(node,'quest_hand',raw)
    Dg5fDebugRecorder._on_teleop_input(node,'landmarks',normalized)
    Dg5fDebugRecorder._on_rosout(node,Log(name='unity_endpoint',msg='Disconnected from Quest',level=20))
    writer.finalize()
    quest=json.loads((writer.run_dir/'quest_hand.jsonl').read_text())
    assert len(quest['points'])==20
    assert quest['source_stamp_sec']==123 and quest['source_stamp_nanosec']==456
    assert 'receipt_monotonic_s' in quest and 'receipt_wall_time_unix_s' in quest
    ros=json.loads((writer.run_dir/'rosout.jsonl').read_text())
    assert ros['message']=='Disconnected from Quest'
    manifest=json.loads((writer.run_dir/'manifest.json').read_text())
    assert manifest['teleop_capture']['counts']==dict(quest_hand=1,landmarks=1,rosout=1)
