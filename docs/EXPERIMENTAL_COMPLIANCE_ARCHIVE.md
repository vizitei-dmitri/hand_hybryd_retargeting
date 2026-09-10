# Архив adaptive compliance

Полное экспериментальное состояние сохранено локально:

- ветка: `archive/adaptive-compliance-2026-09-10`;
- WIP-коммит: `f3ef50ad548257deffe103405a7b4f6492684620`.

В архиве включены dI/dt scaling, contact-aware scaling, URDF/FK limiter,
thumb-index boundary 25/35 mm, dynamic lead, controlled yield, раздельные
actuator/contact relief, adjacent proximity, sustained stall, post-shaper clamp,
диагностика, тесты и документация. На текущей рабочей ветке эти механизмы не
влияют на physical command; contact/FK сохранены только для status/debug.

Потенциально полезной отдельно выглядела robot-space граница thumb-index, но её
следует возвращать и физически проверять независимо от остальных механизмов.

Сравнение без переключения рабочей ветки:

```bash
git diff baseline/current-guard-v2-2026-09-08..archive/adaptive-compliance-2026-09-10 -- \
  src/lerobot_robot_dg5f
git show f3ef50ad548257deffe103405a7b4f6492684620
```

Архив не отправлен в remote.

Baseline восстановлен из ZIP + current_guard_v2.patch и зафиксирован тегом
`baseline/current-guard-v2-2026-09-08`; [происхождение и сравнение](CURRENT_GUARD_V2_BASELINE.md).
В рабочей версии алгоритм эксперимента дополнительно сохранён в
`src/lerobot_robot_dg5f/lerobot_robot_dg5f/experimental_current_guard.py`.
