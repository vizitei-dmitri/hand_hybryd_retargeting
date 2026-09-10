# Рабочий baseline: direct_guarded + current_guard_v2

Эксперименты adaptive compliance/contact-aware tracking отключены от управления
в ROS bridge. Ретаргетинг и его pinch-алгоритмы не менялись.

## Точный reference 08.09, 12:49

Эталон задан пользователем по физическому запуску
`dg5f_debug_2026-09-08_12-49-55.tar.gz`. Исходники восстановлены из
`hand_hybryd_retargeting(3).zip` (локальное имя: `hand_hybryd_retargeting_3.zip`)
с применением `dg5f_current_guard_v2.patch`. Один commit `58a61a5` без этого
патча эталоном не является.

Тег восстановленной рабочей версии: `baseline/current-guard-v2-2026-09-08`.
Новые dataset/debug функции сохранены; тег фиксирует проект с восстановленным
physical control path, а не побайтную копию всего старого проекта.

Результат сравнения пяти control-файлов с ZIP + patch:

| Файл | Соответствие reference |
| --- | --- |
| `current_guard.py` | Побайтное |
| `command_shaper.py` | Побайтное |
| `dg5f.py` | Побайтное |
| `ros_bridge_node.py` | Команды, ARM, recovery, grace/resume сохранены; добавлены только диагностические наблюдатели/события и неактивные параметры |
| `bridge.params.yaml` | Все параметры reference совпадают; дополнительные contact/compliance поля диагностические или отключены |

В отличие от предыдущего приближённого rollback, теперь восстановлены также
reference relief по **конечной абсолютной ошибке**, trip-таймер без нового сброса
после паузы >0.2 s и defaults узла без YAML. В обычном запуске YAML устанавливает
direct mode; при создании узла без YAML остаются исходные defaults
`control_smoothing=true`, `min_send_step_deg=0.20`.

SHA-256 входных файлов:

```text
ZIP   13937b24d01a89bc6dd7944934d2e736cdde541050103a0be778abbbd546813d
patch b69c61ee6872fdf1cd53437f8858a4b96545bb7534ee296dc080e02d1491ea04
```

Reference `current_guard.py` SHA-256:
`8d650ea97b07224384e4d94729a0c3d67b93d2a92a0d8f160a6e89c4a53dde7d`.

При восстановлении: 18 targeted-тестов пройдены, syntax/import и сборка
`lerobot_robot_dg5f` успешны. Проверены также побайтное совпадение трёх файлов,
значения YAML и AST методов отправки/ARM/recovery/grace/resume. Проверки выполнены
в Docker с `--network none` и mock backend. Новый физический тест не проводился.

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
В reference разгрузка означает уменьшение `abs(desired - measured)`, а не только
начальное направление движения. Например, при measured=59°, effective=65° и
hard current цель 60° разрешена, а цель 20° удерживается: её конечная ошибка больше.
При 50 Hz это предел шага команды, не гарантия скорости физического мотора.

## Что отключено

ROS bridge использует неизменённый reference `AdaptiveCurrentGuard` из
`current_guard.py`; в его API вообще нет аргумента `compliance`. Он получает
только токи, measured/effective/desired положения и время. Slope scaling, dynamic
lead, contact gains, граница thumb-index 25/35 mm, adjacent proximity slowdown,
yield и экспериментальный stall не участвуют в вычислении команд или disarm.
`command_bounds` и дополнительный clamp после shaper удалены из `Dg5f.send_action`
и `PositionCommandShaper.step`; обычные joint limits остаются.

`compliance_enabled`, `compliance_yield_enabled`, `compliance_stall_enabled` равны
`false`. Остальные числовые параметры эксперимента оставлены для совместимости
и истории, но не подключены к управлению. Даже старый launch с
`compliance_enabled=true` не включает эксперимент обратно: bridge предупреждает
и остаётся в current_guard_v2. Offline-код эксперимента сохранён побайтно в
`experimental_current_guard.py`; экспериментальные тесты импортируют его оттуда.
Архивная ветка `archive/adaptive-compliance-2026-09-10` также сохранена.

## Только диагностика

`/dg5f/hybrid_contact` и `/dg5f/safety_proximity` продолжают публиковаться.
Human contact weights, расстояния между кончиками, URDF FK расстояния для
measured/effective/desired поз и debug CSV сохранены. FK вызывается в публикации
диагностики, а не в `_send_latest` или current guard. Эти данные не меняют target,
shaper, effective command, токовые коэффициенты или ARM/disarm.
`current_guard_diagnostics.py` только копирует результат reference guard для
debug-полей и считает dI/dt при публикации диагностики (30 Hz по YAML). Его
результаты не используются для задания target или принятия решения о trip.

Исправления SDK, 200 Hz low-level loop, telemetry, recovery, suspend_motion,
fresh-feedback ARM reseed, LeRobotDataset, Quest transport и MuJoCo не откатывались.
Для применения нужен перезапуск уже работающего pipeline; физическую кисть
при внесении этих изменений автоматически не запускали и не ARM-или.
