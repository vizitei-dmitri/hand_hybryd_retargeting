"""Exercise real ROS timers/guard on mock hardware in an isolated ROS domain."""
import asyncio
import json
import time

import numpy as np
import pytest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from rclpy.parameter import Parameter
from std_msgs.msg import Bool
from std_srvs.srv import SetBool

from lerobot_robot_dg5f.constants import JOINT_NAMES
from lerobot_robot_dg5f.ros_bridge_node import Dg5fLeRobotBridge


@pytest.fixture
def bridge(monkeypatch):
    monkeypatch.setenv('ROS_DOMAIN_ID', '94')
    rclpy.init(args=['--ros-args', '-p', 'control_mode:=servo',
                     '-p', 'require_tracking:=false', '-p', 'auto_enable:=true',
                     '-p', 'compliance_enabled:=false', '-p', 'servo_max_velocity_deg_s:=30.0'])
    node = Dg5fLeRobotBridge()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    yield node, executor
    executor.remove_node(node)
    node.destroy_node()
    executor.shutdown()
    rclpy.shutdown()


def target(node, value):
    msg = JointTrajectory()
    msg.joint_names = list(JOINT_NAMES)
    msg.points = [JointTrajectoryPoint(positions=[float(np.deg2rad(value))] * 20)]
    node._on_command(msg)


def arm(node):
    return asyncio.run(node._on_enable(SetBool.Request(data=True), SetBool.Response()))


@pytest.mark.parametrize('input_state', ['absent', 'stale_tracking', 'stale_target', 'tracking_false'])
def test_servo_arm_without_fresh_input_seeds_measured_hold(bridge, monkeypatch, input_state):
    node, _ = bridge
    node._disarm('USER_REQUEST')
    node.set_parameters([Parameter('require_tracking', value=True)])
    measured = np.zeros(20)
    measured[6] = -25
    node._robot.backend._positions = measured.copy()
    if input_state != 'absent':
        node._on_tracking(Bool(data=True))
        target(node, 70)
        if input_state == 'stale_tracking':
            node._last_tracking_time -= 20
        elif input_state == 'stale_target':
            node._last_command_time -= 20
        else:
            node._on_tracking(Bool(data=False))
    sends, records, events = [], [], []
    monkeypatch.setattr(node._robot.backend, 'send_positions', lambda q: sends.append(q.copy()))
    monkeypatch.setattr(node._servo_pub, 'publish', lambda m: records.append(json.loads(m.data)))
    monkeypatch.setattr(node, '_event', lambda name, **kw: events.append(name))
    result = arm(node)
    assert result.success, result.message
    assert node._armed and node._runtime_state() == 'TRACKING_HOLD'
    assert node._disarm_reason == 'NONE'
    assert node._latest_command_deg is None and node._last_command_time is None
    np.testing.assert_array_equal(node._robot.command_shaper.command_pose_deg, measured)
    np.testing.assert_array_equal(node._robot.command_shaper.effective_command(), measured)
    assert len(sends) == 1  # Physical HOLD initialized once, never an old VR target.
    np.testing.assert_array_equal(sends[0], measured)
    assert arm(node).success  # Repeated enable cannot reinitialize the hold pose.
    assert len(sends) == 1
    node._tracking_grace_started -= 20
    node._send_latest()
    node._expire_tracking_grace()
    assert node._armed and node._runtime_state() == 'TRACKING_HOLD'
    assert len(sends) == 1
    assert records[-1]['runtime_state'] == 'TRACKING_HOLD'
    assert records[-1]['command_submitted'] is False
    assert 'DISARMED' not in events


@pytest.mark.parametrize('loss', ['tracking_false', 'tracking_timeout', 'command_timeout'])
def test_arm_hold_arrival_loss_and_return_need_only_one_arm(bridge, monkeypatch, loss):
    node, _ = bridge
    node._disarm('USER_REQUEST')
    node.set_parameters([Parameter('require_tracking', value=True)])
    # Exercise the production velocity setting, independently of the fixture's 30 deg/s.
    controller = node._robot.command_shaper
    controller.max_velocity_deg_s = 120.0
    measured = np.zeros(20)
    measured[6] = -25
    node._robot.backend._positions = measured.copy()
    sends, events = [], []
    monkeypatch.setattr(node._robot.backend, 'send_positions', lambda q: sends.append(q.copy()))
    monkeypatch.setattr(node, '_event', lambda name, **kw: events.append(name))
    assert arm(node).success
    # No tracking: even a valid retarget message must not enter the mailbox.
    target(node, 70)
    assert node._latest_command_deg is None
    node._send_latest()
    assert len(sends) == 1
    node._on_tracking(Bool(data=True))
    node._send_latest()
    assert node._runtime_state() == 'TRACKING_HOLD'  # Need a NEW valid target too.
    target(node, float('nan'))
    assert node._runtime_state() == 'TRACKING_HOLD'
    target(node, 70)
    assert node._runtime_state() == 'ACTIVE'
    np.testing.assert_array_equal(controller.command_pose_deg, measured)
    node._send_latest()
    assert sends[-1][6] == pytest.approx(-24.8)  # First tick: 720 deg/s² from rest.
    held = controller.command_pose_deg.copy()
    physical = sends[-1].copy()
    if loss == 'tracking_false':
        node._on_tracking(Bool(data=False))
    elif loss == 'tracking_timeout':
        node._last_tracking_time -= 20
    else:
        node._last_command_time -= 20
    node._send_latest()
    node._tracking_grace_started -= 20
    node._expire_tracking_grace()
    node._send_latest()
    assert node._armed and node._runtime_state() == 'TRACKING_HOLD'
    np.testing.assert_array_equal(controller.command_pose_deg, held)
    np.testing.assert_array_equal(sends[-1], physical)
    count = len(sends)
    # Tracking recovery cannot reseed from feedback or run the old target.
    node._robot.backend._positions[6] = -50
    monkeypatch.setattr(node._robot, 'begin_arm_blend_from_feedback',
                        lambda **kw: pytest.fail('Reseed on tracking recovery'))
    node._on_tracking(Bool(data=True))
    node._send_latest()
    assert node._runtime_state() == 'TRACKING_HOLD'
    assert len(sends) == count
    target(node, -60)
    assert node._armed and node._runtime_state() == 'ACTIVE'
    node._send_latest()
    assert sends[-1][6] == pytest.approx(physical[6] - .2)
    assert np.max(np.abs(sends[-1] - physical)) <= 2 + 1e-9
    assert events.count('ARM_REQUESTED') == events.count('ARMED') == 1
    assert events.count('TRACKING_RECOVERED') == 2
    assert 'DISARMED' not in events


@pytest.mark.parametrize('status,reason', [
    ({'transport_connected': False}, 'transport not connected'),
    ({'control_thread_alive': False}, 'control thread is not alive'),
    ({'telemetry_valid': False}, 'telemetry is stale'),
    ({'temperature_safe': False}, 'temperature is unsafe'),
    ({'system_started': False}, 'system is not started'),
    ({'recovery_required': True}, 'recovery is required'),
    ({'motion_ready': False, 'motion_ready_reason': 'MOTION_RESULT_ERROR'}, 'MOTION_RESULT_ERROR'),
    ({'last_motion_result': 500}, 'SDK_COMMAND_REJECTED'),
    ({'last_position_sample_age_ms': 10000}, 'PHYSICAL_POSE_STALE'),
])
def test_servo_arm_without_tracking_still_checks_hardware(bridge, monkeypatch, status, reason):
    node, _ = bridge
    node._disarm('USER_REQUEST')
    node.set_parameters([Parameter('require_tracking', value=True)])
    original = node._robot.backend.read_control_status
    monkeypatch.setattr(node._robot.backend, 'read_control_status', lambda: {**original(), **status})
    monkeypatch.setattr(node._robot.backend, 'send_positions', lambda q: pytest.fail('Unsafe ARM send'))
    result = arm(node)
    assert not result.success and reason in result.message
    assert not node._armed


def test_legacy_arm_still_requires_tracking(bridge, monkeypatch):
    node, _ = bridge
    node._disarm('USER_REQUEST')
    node._control_mode = 'legacy'
    node.set_parameters([Parameter('require_tracking', value=True)])
    monkeypatch.setattr(node._robot.backend, 'send_positions', lambda q: pytest.fail('Legacy ARM sent'))
    result = arm(node)
    assert not result.success and result.message == 'ARM FAILED: TRACKING_TIMEOUT'


def test_servo_arm_rejected_hold_submission_does_not_authorize_output(bridge, monkeypatch):
    node, _ = bridge
    node._disarm('USER_REQUEST')
    node.set_parameters([Parameter('require_tracking', value=True)])
    def reject(q):
        raise RuntimeError('SDK rejected measured hold')
    monkeypatch.setattr(node._robot.backend, 'send_positions', reject)
    result = arm(node)
    assert not result.success
    assert 'SDK rejected measured hold' in result.message
    assert not node._armed


def test_servo_arm_rejects_invalid_measured_pose_without_tracking(bridge, monkeypatch):
    node, _ = bridge
    node._disarm('USER_REQUEST')
    node.set_parameters([Parameter('require_tracking', value=True)])
    node._robot.backend._positions[6] = float('nan')
    monkeypatch.setattr(node._robot.backend, 'send_positions', lambda q: pytest.fail('Invalid pose send'))
    result = arm(node)
    assert not result.success and 'Initial pose contains NaN or infinity' in result.message
    assert not node._armed


def test_timer_streams_independently_of_ten_hz_input_and_logs_every_send(bridge, monkeypatch):
    node, executor = bridge
    sends, records = [], []
    original_send = node._robot.backend.send_positions
    def send(q):
        sends.append((time.monotonic(), q.copy()))
        original_send(q)
    monkeypatch.setattr(node._robot.backend, 'send_positions', send)
    node.create_subscription(String, '/dg5f/lerobot/servo_state',
                             lambda msg: records.append(json.loads(msg.data)), 100)
    target(node, 30)
    assert not sends  # Target callback must never send to the backend.
    node.create_timer(0.1, lambda: target(node, 30))
    deadline = time.monotonic() + 1.5
    while time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.01)
    assert 70 <= len(sends) <= 105
    rate = (len(sends) - 1) / (sends[-1][0] - sends[0][0])
    assert 50 <= rate <= 70
    # Real timer jitter changes dt; speed remains <=30 deg/s.
    assert all(row['max_step_deg'] <= 30 * row['limiter_dt'] + 1e-8 for row in records)
    assert len(records) >= len(sends) - 3
    assert all(len(records[-1][key]) == 20 for key in
               ('q_target', 'q_cmd', 'q_measured', 'tracking_error'))
    print(f' mock servo: {rate:.3f} Hz, {len(sends)} full-pose sends, '
          f'max step {max(row["max_step_deg"] for row in records):.6f} deg')


def test_overcurrent_is_local_and_never_disarms(bridge, monkeypatch):
    node, _ = bridge
    sends = []
    original = node._robot.get_diagnostics
    current = np.zeros(20)
    current[6] = 700  # Hard threshold, below sustained trip.
    def diagnostics():
        result = original()
        result.update(measured_pos=np.zeros(20), measured_current=current.copy())
        return result
    monkeypatch.setattr(node._robot, 'get_diagnostics', diagnostics)
    monkeypatch.setattr(node._robot.backend, 'send_positions', lambda q: sends.append(q.copy()))
    now = time.monotonic()
    target(node, 60)
    node._send_servo_tick(now)
    assert sends[-1][6] == 0
    assert sends[-1][10] == pytest.approx(0.2)  # Unloaded joint accelerates freely.
    current[6] = 900
    node._send_servo_tick(now + 0.02)
    count = len(sends)
    node._send_servo_tick(now + 0.08)
    assert node._armed
    assert node._disarm_reason == 'NONE'
    assert len(sends) == count + 1
    assert sends[-1][6] == 0
    assert sends[-1][10] > sends[-2][10]


def test_latest_target_replaces_previous_and_stale_input_holds(bridge, monkeypatch):
    node, _ = bridge
    sends = []
    monkeypatch.setattr(node._robot.backend, 'send_positions', lambda q: sends.append(q.copy()))
    target(node, 80)
    target(node, -80)
    assert not sends
    node._send_latest()
    assert sends[-1][6] == pytest.approx(-0.2)
    node._last_command_time = time.monotonic() - 1
    node._send_latest()
    assert node._armed
    assert node._runtime_state() == 'TRACKING_HOLD'
    assert len(sends) == 1


def test_tracking_error_warning_does_not_disarm_or_reseed(bridge, monkeypatch):
    node, _ = bridge
    sends, warnings = [], []
    monkeypatch.setattr(node._robot.backend, 'send_positions', lambda q: sends.append(q.copy()))
    monkeypatch.setattr(type(node.get_logger()), 'warning', lambda self, msg, **kw: warnings.append(msg))
    now = time.monotonic()
    target(node, 60)
    for index in range(65):
        node._send_servo_tick(now + index / 60)
    assert node._armed
    assert len(sends) == 65
    assert sends[-1][6] == pytest.approx(32.1)
    assert any('>10 deg for >=300 ms' in msg for msg in warnings)


def test_disarmed_bridge_never_submits(bridge, monkeypatch):
    node, _ = bridge
    target(node, 60)
    node._disarm('USER_REQUEST')
    monkeypatch.setattr(node._robot.backend, 'send_positions',
                        lambda q: pytest.fail('Motion while disarmed'))
    node._send_latest()


def test_tracking_lost_over_15s_then_recovers_without_reseed(bridge):
    from rclpy.parameter import Parameter
    from std_msgs.msg import Bool
    node, _ = bridge
    node.set_parameters([Parameter('require_tracking', value=True)])
    node._tracking_ok = True
    node._last_tracking_time = time.monotonic()
    target(node, 70)
    node._send_latest()
    finger = node._current_guard._object_contact.fingers['index']
    finger.reason = 'OPERATOR_OPENING'
    finger.resume_offset = np.array([0., 100., 0.])
    node._on_tracking(Bool(data=False))
    held = node._robot.command_shaper.effective_command()
    assert node._tracking_grace_active and node._armed
    node._send_latest()
    np.testing.assert_array_equal(node._robot.command_shaper.effective_command(), held)
    node._tracking_grace_started -= 20
    node._expire_tracking_grace()
    node._send_latest()
    assert node._armed and node._runtime_state() == 'TRACKING_HOLD'
    # Targets received while tracking=false cannot replace the held target.
    target(node, -30)
    assert node._latest_command_deg[6] == pytest.approx(70)
    # Even very different feedback must not re-anchor the persistent trajectory.
    node._robot.backend._positions[:] = 0
    node._robot.backend._positions[6] = -50
    node._on_tracking(Bool(data=True))
    target(node, 70)
    assert not node._tracking_grace_active and node._armed
    np.testing.assert_array_equal(node._robot.command_shaper.command_pose_deg, held)
    node._send_latest()
    continued = node._robot.command_shaper.command_pose_deg
    assert 0 <= continued[6] - held[6] <= 0.5 + 1e-9


def test_manual_rearm_drops_old_offset_and_moves_from_feedback(bridge):
    from std_srvs.srv import SetBool
    node, executor = bridge
    node._disarm('USER_REQUEST')
    node._current_guard._object_contact.fingers['index'].resume_offset = np.array([0., 80., 0.])
    node._robot.backend._positions[6] = -50
    target(node, 70)
    task = executor.create_task(node._on_enable, SetBool.Request(data=True), SetBool.Response())
    executor.spin_until_future_complete(task, timeout_sec=2)
    assert task.result().success
    assert node._current_guard._object_contact.fingers['index'].resume_offset is None
    previous = node._robot.command_shaper.effective_command()[6]
    node._send_latest()
    assert node._robot.command_shaper.effective_command()[6] >= previous


def test_direction_error_contains_every_stage_and_reason(bridge, monkeypatch):
    node, _ = bridge
    errors, events = [], []
    monkeypatch.setattr(type(node.get_logger()), 'error', lambda self,msg,**kw:errors.append(msg))
    monkeypatch.setattr(node, '_event', lambda name,**kw:events.append((name,kw)))
    initial = np.zeros(20); initial[6] = -50
    node._robot.command_shaper.reset(initial)
    node._robot.backend._positions = initial.copy()
    # Inject a backwards contact-stage result to exercise the retained
    # direction guard; servo FREE no longer applies legacy resume offsets.
    contact = node._current_guard._object_contact
    original_update = contact.update
    def backwards_contact(**kwargs):
        decision = original_update(**kwargs)
        decision.target_deg[6] = -55
        decision.limited_mask[6] = True
        return decision
    monkeypatch.setattr(contact, 'update', backwards_contact)
    target(node,-49)
    node._send_latest()
    assert node._robot.command_shaper.effective_command()[6] == -50
    telemetry = node._robot.command_shaper.telemetry
    assert not telemetry['direction_violation'][6]
    assert telemetry['post_contact_direction_violation'][6]
    events = [event for event in events if event[0] == 'COMMAND_DIRECTION_VIOLATION_BLOCKED']
    assert len(events) == 1
    assert events[0][1]['reason'] == 'BLOCKED_CONTACT_OFFSET_REVERSAL'
    assert events[0][1]['joint'] == 'rj_dg_2_3'
    for key in ('target','previous_q_cmd','raw_servo_step','new_q_cmd',
                'post_compliance_cmd','post_guard_cmd','low_level_cmd','measured'):
        assert key in events[0][1]
        assert key in errors[0]


def test_explicit_recovery_stays_disarmed_then_accepts_fresh_rearm(bridge, monkeypatch):
    from std_srvs.srv import Trigger, SetBool
    node, executor = bridge
    node._backend_name = 'tesollo'  # Service gate only; backend remains mock.
    def recover(timeout):
        node._robot.backend._positions[:] = 0
        node._robot.backend._positions[6] = -50
        return node._robot.backend._positions.copy()
    monkeypatch.setattr(node._robot.backend, 'recover', recover, raising=False)
    target(node,70)
    node._current_guard._object_contact.fingers['index'].resume_offset = np.array([0., 80., 0.])
    task=executor.create_task(node._on_recover,Trigger.Request(),Trigger.Response())
    executor.spin_until_future_complete(task,timeout_sec=2)
    assert task.result().success
    assert not node._armed
    assert node._latest_command_deg is None
    assert node._current_guard._object_contact.fingers['index'].resume_offset is None
    assert node._robot.command_shaper.effective_command()[6] == -50
    target(node,70)
    task=executor.create_task(node._on_enable,SetBool.Request(data=True),SetBool.Response())
    executor.spin_until_future_complete(task,timeout_sec=2)
    assert task.result().success
    node._send_latest()
    assert node._robot.command_shaper.effective_command()[6] > -50


def test_stalled_index_physical_lead_is_bounded_and_other_joints_move(bridge, monkeypatch):
    from lerobot_robot_dg5f.current_guard import ComplianceConfig
    node, _ = bridge
    node._current_guard.compliance = ComplianceConfig()
    initial = np.zeros(20); initial[6] = 58
    node._robot.command_shaper.reset(initial)
    measured = initial.copy()
    current = np.zeros(20)
    original = node._robot.get_diagnostics
    sends, events = [], []
    def diagnostics():
        result = original()
        result.update(measured_pos=measured.copy(), measured_current=current.copy())
        return result
    monkeypatch.setattr(node._robot, 'get_diagnostics', diagnostics)
    monkeypatch.setattr(node._robot.backend, 'send_positions', lambda q:sends.append(q.copy()))
    monkeypatch.setattr(node, '_event', lambda name,**kw:events.append((name,kw)))
    target(node, 71.5)
    start = time.monotonic()
    for tick in range(100):
        if sends:
            measured[:] = sends[-1]  # All joints except index follow normally.
            measured[6] = 58
        current[6] = min(1200, tick * 20)
        node._send_servo_tick(start + tick / 60)
        budget = node._compliance_diagnostics['joint_lead_budget_deg']
        assert abs(sends[-1][6] - 58) <= budget[6] + 1e-8
        assert node._armed and node._sdk_block_reason is None
    assert sends[-1][10] > sends[60][10]  # index > old threshold; middle still moves
    assert any(name == 'CURRENT_LOAD_THRESHOLD' for name,_ in events)
    assert not any(name in ('DISARMED','CURRENT_GUARD_TRIP','STALL_GUARD_TRIP') for name,_ in events)
    # Existing large lead must also shrink, even when q_cmd has no forward step.
    node._robot.command_shaper.command_pose_deg[6] = 71
    node._send_servo_tick(start + 101 / 60)
    assert sends[-1][6] == pytest.approx(59)
    assert node._robot.command_shaper.command_pose_deg[6] != measured[6]
    assert node._robot.command_shaper.telemetry['physical_limit_reason'][6] == 'LEAD_BUDGET_YIELD_OR_HOLD'


def test_sdk_fault_latches_backend_without_global_disarm_or_automatic_reset(bridge, monkeypatch):
    from std_srvs.srv import SetBool
    node, executor = bridge
    events = []
    original = node._robot.get_diagnostics
    broken = [True]
    def diagnostics():
        result = original()
        if broken[0]:
            result.update(last_motion_result=500, motion_ready=False,
                          motion_ready_reason='MOTION_RESULT_ERROR')
        return result
    monkeypatch.setattr(node._robot, 'get_diagnostics', diagnostics)
    monkeypatch.setattr(node, '_event', lambda name,**kw:events.append((name,kw)))
    target(node, 70)
    node._publish_state()
    assert node._armed and node._runtime_state() == 'SDK_ERROR'
    assert events[0][0] == 'SDK_ERROR' and events[0][1]['last_motion_result'] == 500
    assert not any(name == 'DISARMED' for name,_ in events)
    broken[0] = False  # A healthy snapshot cannot implicitly clear the latch.
    monkeypatch.setattr(node._robot.backend, 'send_positions', lambda q:pytest.fail('send after fault'))
    node._send_latest()
    task=executor.create_task(node._on_enable,SetBool.Request(data=True),SetBool.Response())
    executor.spin_until_future_complete(task,timeout_sec=2)
    assert not task.result().success
    assert node._sdk_block_reason


def test_hold_emits_full_diagnostics_and_resumes_on_new_command_without_tracking_gate(bridge, monkeypatch):
    node, executor = bridge
    records = []
    node.create_subscription(String, '/dg5f/lerobot/servo_state', lambda m:records.append(json.loads(m.data)),100)
    target(node, 70)
    node._send_latest()
    held = node._robot.command_shaper.command_pose_deg.copy()
    node._last_command_time -= 20
    node._send_latest()
    for _ in range(10):
        executor.spin_once(timeout_sec=0.005)
    hold = [r for r in records if r['runtime_state'] == 'TRACKING_HOLD'][-1]
    for key in ('q_target','previous_q_cmd','q_cmd','q_measured','raw_servo_step',
                'post_contact_cmd','post_guard_cmd','low_level_cmd','current_ma','total_current_ma',
                'joint_lead_budget_deg','actual_command_lead_deg','object_contact_state',
                'object_contact_reason','tracking_ok','tracking_loss_reason',
                'last_quest_age_ms','last_landmarks_age_ms','last_retarget_age_ms'):
        assert key in hold
    assert hold['command_submitted'] is False
    target(node, -30)
    assert node._runtime_state() == 'ACTIVE'
    node._send_latest()
    resumed = node._robot.command_shaper.command_pose_deg
    assert np.max(np.abs(resumed-held)) <= 0.5+1e-8


def test_accelerated_reverse_survives_real_guard_and_tracking_hold(bridge, monkeypatch):
    node, _ = bridge
    s = node._robot.command_shaper
    s.max_velocity_deg_s = 120
    events, records = [], []
    monkeypatch.setattr(node, '_event', lambda name, **kw: events.append(name))
    monkeypatch.setattr(node._servo_pub, 'publish', lambda m: records.append(json.loads(m.data)))
    now = time.monotonic()
    target(node, 70)
    for tick in range(12):
        node._send_servo_tick(now + tick / 60)
    assert s.velocity_deg_s[6] == pytest.approx(120)
    target(node, -70)
    velocities = []
    previous_v = 120
    for tick in range(12, 34):
        node._send_servo_tick(now + tick / 60)
        v = s.velocity_deg_s[6]
        assert abs(v - previous_v) <= 12 + 1e-6
        velocities.append(v)
        previous_v = v
    assert velocities[0] == pytest.approx(108)
    assert any(abs(v) < 1e-6 for v in velocities)
    assert min(velocities) == pytest.approx(-120)
    assert records[12]['trajectory_braking'][6]
    assert not any(name in ('COMMAND_MOVED_AWAY_FROM_TARGET',
                           'COMMAND_DIRECTION_VIOLATION_BLOCKED') for name in events)
    held = s.command_pose_deg.copy()
    node._enter_tracking_grace(now + 34 / 60)
    assert not np.any(s.velocity_deg_s)
    node._publish_servo_hold(now + 35 / 60)
    assert records[-1]['commanded_velocity_deg_s'] == [0.] * 20
    np.testing.assert_array_equal(s.command_pose_deg, held)
    # Simulated monotonic clock so the recovery target follows the HOLD timestamp.
    monkeypatch.setattr(node, '_now_seconds', lambda: now + 36 / 60)
    target(node, 70)
    assert node._runtime_state() == 'ACTIVE'
    node._send_servo_tick(now + 36 / 60)
    assert s.velocity_deg_s[6] == pytest.approx(12)
    assert s.command_pose_deg[6] - held[6] == pytest.approx(.2)
