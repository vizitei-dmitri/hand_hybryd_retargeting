# Рабочий baseline: direct_guarded + current_guard_v2

Эксперименты adaptive compliance/contact-aware tracking отключены от управления
в ROS bridge. Ретаргетинг и его pinch-алгоритмы не менялись.

## Активный путь команд

```text
Quest → прежний retarget → joint limits / rj_dg_5_1 = 0
      → current_guard_v2 → PositionCommandShaper direct_guarded
      → шаг не более 5° → DG5F / штатные защиты DGSDK
```

Параметры в `src/lerobot_robot_dg5f/config/bridge.params.yaml`:

| Параметр | Значение |
| --- | --- |
| `control_smoothing` | `false` |
| `max_direct_step_deg` | `5.0` |
| Ток одного сустава: soft / hard / trip | 350 / 650 / 850 mA |
| Сумма модулей токов: soft / hard / trip | 600 / 850 / 1050 mA |
| `current_guard_trip_hold_s` | 0.04 s |
| `current_guard_release_tau_s` | 0.20 s |
| `startup_blend_s` | 0.70 s |
| `tracking_grace_s` | 15 s |
| `tracking_resume_blend_s` | 1.0 s |

Ниже soft токовый guard не вносит дополнительного ограничения после отпускания
предыдущей нагрузки. Между soft и hard ограничивается дальнейшее нагружение.
При hard команда в сторону увеличения нагрузки удерживается на effective pose,
без автоматического отступления к measured pose. Движение оператора в сторону
разгрузки проходит быстро, но сохраняет предел 5°/шаг и emergency trip.
При 50 Hz это предел шага команды, не гарантия скорости физического мотора.

## Что отключено

ROS bridge создаёт `AdaptiveCurrentGuard(..., compliance=None)` и передаёт ему
только токи, measured/effective/desired положения и время. Slope scaling, dynamic
lead, contact gains, граница thumb-index 25/35 mm, adjacent proximity slowdown,
yield и экспериментальный stall не участвуют в вычислении команд или disarm.
`command_bounds` и дополнительный clamp после shaper удалены из `Dg5f.send_action`
и `PositionCommandShaper.step`; обычные joint limits остаются.

`compliance_enabled`, `compliance_yield_enabled`, `compliance_stall_enabled` равны
`false`. Остальные числовые параметры эксперимента оставлены для совместимости
и истории, но не подключены к управлению. Даже старый launch с
`compliance_enabled=true` не включает эксперимент обратно: bridge предупреждает
и остаётся в current_guard_v2. Offline-код эксперимента сохранён, но ROS его не включает.

## Только диагностика

`/dg5f/hybrid_contact` и `/dg5f/safety_proximity` продолжают публиковаться.
Human contact weights, расстояния между кончиками, URDF FK расстояния для
measured/effective/desired поз и debug CSV сохранены. FK вызывается в публикации
диагностики, а не в `_send_latest` или current guard. Эти данные не меняют target,
shaper, effective command, токовые коэффициенты или ARM/disarm.

Исправления SDK, 200 Hz low-level loop, telemetry, recovery, suspend_motion,
fresh-feedback ARM reseed, LeRobotDataset, Quest transport и MuJoCo не откатывались.
Для применения нужен перезапуск уже работающего pipeline; физическую кисть
при внесении этих изменений автоматически не запускали и не ARM-или.
