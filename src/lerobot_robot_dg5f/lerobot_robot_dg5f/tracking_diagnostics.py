"""Passive upstream freshness observations; never changes retargeting or watchdogs."""

import numpy as np

from .health import age_ms


class TrackingDiagnostics:
    def __init__(self):
        self.streams = {name: dict(received=None, stamp=None, stamp_changed=None,
                                   valid=False, count=0)
                        for name in ('quest', 'landmarks')}

    def observe(self, name, points, stamp, now):
        stream = self.streams[name]
        stream['received'] = now
        stream['count'] += 1
        values = np.asarray(points, dtype=float)
        stream['valid'] = values.shape == (21, 3) and bool(np.all(np.isfinite(values)))
        if stamp > 0:
            if stream['stamp'] is None or stamp > stream['stamp']:
                stream['stamp_changed'] = now
            stream['stamp'] = stamp

    def snapshot(self, *, now, retarget_time, tracking_ok, tracking_time, timeout):
        quest, landmarks = self.streams['quest'], self.streams['landmarks']
        quest_age = age_ms(quest['received'], now)
        landmarks_age = age_ms(landmarks['received'], now)
        retarget_age = age_ms(retarget_time, now)
        stamp_age = age_ms(landmarks['stamp_changed'], now)
        fresh_quest = quest['received'] is not None and quest_age <= timeout * 1000
        fresh_landmarks = landmarks['received'] is not None and landmarks_age <= timeout * 1000
        detail = 'NONE'
        if fresh_quest and not quest['valid']:
            reason, detail = 'INVALID_HAND', 'INVALID_QUEST_LANDMARKS'
        elif landmarks['received'] is None:
            reason = 'NO_LANDMARKS'
            detail = 'QUEST_RECEIVED_NO_ADAPTER_OUTPUT' if fresh_quest else 'NO_QUEST_HAND_INPUT'
        elif not fresh_landmarks:
            reason = 'STALE_LANDMARKS'
            detail = 'LANDMARK_ADAPTER_STALE' if fresh_quest else 'QUEST_INPUT_STALE'
        elif not landmarks['valid']:
            reason, detail = 'INVALID_HAND', 'INVALID_NORMALIZED_LANDMARKS'
        elif landmarks['stamp'] is not None and stamp_age > timeout * 1000:
            reason, detail = 'STALE_LANDMARKS', 'SOURCE_TIMESTAMP_STALLED'
        elif retarget_time is None or retarget_age > timeout * 1000:
            reason = 'RETARGET_STALE'
        elif tracking_time is None or age_ms(tracking_time, now) > timeout * 1000:
            reason = 'TRACKING_STATUS_STALE'
        elif not tracking_ok:
            reason = 'UPSTREAM_TRACKING_FALSE'
        else:
            reason = 'NONE'
        return dict(
            last_quest_received_monotonic_s=quest['received'],
            last_landmarks_received_monotonic_s=landmarks['received'],
            last_successful_retarget_monotonic_s=retarget_time,
            last_quest_age_ms=quest_age,
            last_landmarks_age_ms=landmarks_age,
            last_retarget_age_ms=retarget_age,
            landmarks_stamp_age_ms=stamp_age,
            quest_header_stamp_s=quest['stamp'],
            landmarks_header_stamp_s=landmarks['stamp'],
            quest_message_count=quest['count'], landmarks_message_count=landmarks['count'],
            tracking_ok=bool(tracking_ok), tracking_loss_reason=reason,
            tracking_loss_detail=detail,
            # Missing hand messages alone cannot prove a broken TCP connection.
            unity_connection_status='UNKNOWN',
        )
