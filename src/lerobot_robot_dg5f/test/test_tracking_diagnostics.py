import numpy as np

from lerobot_robot_dg5f.tracking_diagnostics import TrackingDiagnostics


def status(monitor, now, command=1, tracking=True):
    return monitor.snapshot(now=now, retarget_time=command, tracking_ok=tracking,
                            tracking_time=now, timeout=.35)


def feed(monitor, now, stamp=None):
    for name in ('quest', 'landmarks'):
        monitor.observe(name, np.zeros((21,3)), now if stamp is None else stamp, now)


def test_loss_of_input_and_recovery_are_distinct_from_tesollo_disconnect():
    monitor = TrackingDiagnostics()
    assert status(monitor,1)['tracking_loss_reason'] == 'NO_LANDMARKS'
    feed(monitor,1)
    assert status(monitor,1)['tracking_loss_reason'] == 'NONE'
    lost = status(monitor,1.5,tracking=False)
    assert lost['tracking_loss_reason'] == 'STALE_LANDMARKS'
    assert lost['tracking_loss_detail'] == 'QUEST_INPUT_STALE'
    assert lost['unity_connection_status'] == 'UNKNOWN'
    feed(monitor,2)
    assert status(monitor,2,command=2)['tracking_loss_reason'] == 'NONE'


def test_live_quest_but_missing_adapter_and_retarget_outputs():
    monitor=TrackingDiagnostics()
    feed(monitor,1)
    monitor.observe('quest',np.zeros((21,3)),1.5,1.5)
    assert status(monitor,1.5)['tracking_loss_detail'] == 'LANDMARK_ADAPTER_STALE'
    feed(monitor,2)
    assert status(monitor,2)['tracking_loss_reason'] == 'RETARGET_STALE'


def test_invalid_hand_and_stagnant_source_stamp():
    monitor=TrackingDiagnostics()
    feed(monitor,1)
    monitor.observe('quest',np.zeros((20,3)),1.1,1.1)
    assert status(monitor,1.1)['tracking_loss_reason'] == 'INVALID_HAND'
    feed(monitor,1.5,stamp=1)
    result=status(monitor,1.5,command=1.5)
    assert result['tracking_loss_detail'] == 'SOURCE_TIMESTAMP_STALLED'
    assert result['last_landmarks_age_ms'] == 0
    assert result['landmarks_stamp_age_ms'] == 500
    feed(monitor,1.6)
    assert status(monitor,1.6,command=1.6)['tracking_loss_reason'] == 'NONE'
