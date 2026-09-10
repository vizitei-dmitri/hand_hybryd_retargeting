# Adaptive compliance для DG5F

> Архив эксперимента. Описанные ниже contact/slope/lead/yield/stall ограничения
> **отключены от активного ROS control path**, включая output clamp после shaper.
> Сейчас работает [direct_guarded + current_guard_v2](CURRENT_GUARD_V2_BASELINE.md).
> Параметры и offline-тесты эксперимента сохранены для истории; они не означают,
> что эти алгоритмы управляют физической кистью. Contact/FK — только диагностика.

Реализация в ветке `feature/lerobot-dataset-recorder`. Это дополнительный
software-слой, изменяющий **референс положения**, а не torque controller.
Firmware, DGSDK и внутренние защиты Tesollo не изменены.
Параметры являются стартовыми настройками: поведение на физической кисти
нужно проверить отдельно. Mock-тест не доказывает безопасность реального захвата.

Обновление robot-space contact: MANO/hybrid означает намерение pinch, а не
физический контакт DG5F. Близость теперь определяется по существующей URDF FK
для measured, effective и desired поз. Для thumb-index начало замедления —
35 mm, контактная граница reference — 25 mm (оба значения в config).

После физического теста добавлен controlled yield и разделены actuator/contact
relief. Thumb-index 25/35 mm и bounded reference сохранены. Новые настройки yield
и мягкого отклика ещё требуют отдельной физической проверки оператором.

## Где работает защита

```text
Quest landmarks -> прежний Vector / DexPilot / Hybrid -> q_desired
                      |                                  |
                      +-> hybrid_contact                 +-> MuJoCo (как прежде)
                      +-> safety_proximity               |
                                  |                      v
measured position/current -> AdaptiveCurrentGuard -> разрешённый target + bounds
                                                      |
                                              PositionCommandShaper
                                                      |
                                 q_effective (последняя принятая команда)
                                                      |
                                              прежний DGSDK backend
```

`q_desired` — исходная VR-команда; `q_effective` — последний принятый setpoint,
а не измеренное положение; `q_measured` — обратная связь DG5F.
Угол внутри guard/shaper — в градусах, ток — в mA. ROS JointState/Trajectory
по-прежнему используют радианы. Dataset recorder не менялся: action берётся
из финального `commanded_joint_states`, observation — из обратной связи.
MuJoCo продолжает показывать исходную команду: под нагрузкой его поза может
отличаться от physical/effective pose, и это ожидаемо.

## Как вычисляется разрешённое движение

Для каждого сустава определяются независимые, диагностируемые составляющие:

- `joint_current_scale`: линейно от 1 при soft current до 0 при hard current;
- `global_current_scale`: то же для суммы модулей токов всех 20 приводов;
- `joint_slope_scale`, `global_slope_scale`: снижение при положительном dI/dt;
- `joint_contact_scale`: близость кончиков вместе с локальным током и dI/dt.

Общая шкала:

```text
s = min(joint_current_scale, global_current_scale,
        joint_slope_scale, global_slope_scale, joint_contact_scale)
```

Hard current/global hard и trip действуют без новой задержки. Существующая
токовая шкала сохраняет немедленный attack и `current_guard_release_tau_s=0.20`.
Thumb-index также сохраняет прежний immediate attack/release и reference bound.
Для мягкого slope/sustained-load и остальных contact-пар теперь отдельные
фильтры: `soft_attack_tau_s=0.03`, `soft_release_tau_s=0.06`, без каскадного
применения общего 0.2 s фильтра. При восстановлении до >0.995 шкала округляется
до 1, чтобы не оставлять длительный незначимый ACTIVE-хвост.

Оценка dI/dt: медиана последних 3 измерений модулей токов, линейная регрессия
на последних 5 отфильтрованных отсчётах, затем EMA с tau=0.05 s. Единицы —
mA/s. Производная не вычисляется из одного шумного скачка. Суммарный slope
равен сумме поканальных оценок. Его упреждающее глобальное действие применяется,
когда растут токи как минимум двух пальцев либо суммарный ток уже достиг soft.
Один нагружающийся мотор не должен заранее тормозить независимый мизинец.
Абсолютная защита по total current действует независимо от этого условия.

Для thumb-index `d = min(robot_measured_distance, robot_effective_distance)`.
Effective учитывается, чтобы reference не ушёл внутрь границы раньше, чем
физическая рука его догонит. Desired distance используется для направления
движения, но не считается доказательством близости физической кисти.

```text
x = clip((slowdown_start_mm - d_mm) / (slowdown_start_mm - contact_distance_mm), 0, 1)
w = x*x*(3 - 2*x)
```

Для thumb-middle/ring/little proximity-профиль применяется к robot distance
и умножается на human intent пары. Adjacent pairs вообще не используют
human proximity как основание для торможения: robot proximity — диагностика,
а scale уменьшается только при дополнительном токе/росте тока. Новых жёстких
геометрических границ для этих шести пар нет. При малом токе их scale=1.
Локальная нагрузка `L` — максимум по суставам двух пальцев из:

```text
clip((I_joint - 100) / (350 - 100), 0, 1)
joint_slope_risk
```

Тогда для суставов, продолжающих сближение:

```text
thumb_index_scale = (1 - w) * (1 - w*L)
other_pair_scale = 1 - other_contact_max_reduction*w*L   # reduction=0.5
```

При thumb-index distance >=35 mm и малом токе contact scale=1, даже если
человек уже сомкнул пальцы. При 30 mm scale=0.5 без нагрузки; при 28 mm —
0.216, и растущий ток дополнительно уменьшает его. При 25 mm closing scale=0:
не требуется ждать перегрузки мотора, чтобы перестать требовать 7 mm.
Это эмпирический защитный диапазон для этого прототипа, не измерение силы.

Дополнительно общий closing-шаг thumb/index ограничивается оставшимся
расстоянием effective reference до 25 mm через Jacobian. Существующая FK
проверяет closing-only кандидат без зачёта «компенсирующего» opening другого
сустава. При нарушении границы шаг уменьшается до 5 проверок, затем удерживается.
Это проверка кандидата команды, не модель динамики или всех столкновений.
Если effective уже глубже 25 mm, он не отбрасывается обратно: дальнейшее
закрытие блокируется, opening разрешается. Финальные bounds/shaper сохранены.

Для ограничиваемого closing-сустава:

```text
delta = q_desired - q_effective
step_limit = min(max_direct_step_deg*s, abs(delta)*s)

lead_budget = lead_min_deg + (lead_max_deg - lead_min_deg)*s
remaining_lead = max(0, lead_budget - abs(q_effective - q_measured))

# При наличии нагрузки дополнительно:
step_limit = min(step_limit, remaining_lead)

q_allowed = q_effective + clip(delta, -step_limit, +step_limit)
```

Lead budget ограничивает дальнейшее накопление ошибки команды относительно
feedback. Мягкий contact остальных шести пар больше не уменьшает этот бюджет:
сам по себе thumb-ring при 180–220 mA и ошибке 5° не должен получать scale=0.
Токовая/глобальная защита, slope и подтверждённая нагрузка остаются независимыми
причинами ограничения. Разница desired/effective допустима.

### Controlled yield при внешнем препятствии

Вход: ток сустава >=240 mA, ошибка effective/measured >=4°, отфильтрованная
скорость feedback <=1°/s непрерывно 0.25 s. До подтверждения reference не
отступает; при ожидании не разрешается наращивать lead сверх текущего бюджета.
После подтверждения sustained risk `clip(I_joint/current_soft, 0, 1)` уменьшает
существующий динамический lead budget через мягкую шкалу. При 300 mA и
soft=350 mA установившийся budget — около 2°.

```text
yield_target = measured + sign(effective - measured)*lead_budget
yield_step <= yield_rate_deg_s*dt  # 5°/s; при 50 Hz до 0.1° за кадр
```

Шаг направлен только к yield_target, не перескакивает через него и measured.
Подтверждённый режим сохраняется и ниже порога входа 4°, чтобы error мог
снизиться до 2°. Снятие нагрузки, движение feedback или actuator relief
снимают подтверждение. При hard current остаётся freeze/trip, а не yield.
Новый yield проходит те же final bounds и thumb-index robot-space boundary.
Он может быть ограничен этой границей, если уменьшение actuator error
геометрически сближает thumb-index. Это не выключение position controller.

## Closing, spread и relief

Read-only `ContactKinematics` читает ту же URDF и вычисляет позиции кончиков,
их аналитические Jacobian и производную расстояния пары по каждому углу.
Знак `d(distance)/d(q_j) * delta_j` определяет, сближает ли сустав кончики.
Проверка выполняется в текущей effective pose. Поэтому spread/lateral joint
ограничивается не по номеру, а только если действительно уменьшает расстояние
соответствующей пары. Thumb-index влияет на thumb/index, middle-ring — на
middle/ring; отдельная глобальная токовая защита при этом остаётся общей.

Два разных понятия relief, без объединения в один boolean:

```text
actuator_relief = (q_desired - q_effective) * (q_effective - q_measured) < 0
contact_relief = robot-space движение увеличивает расстояние соответствующей пары
```

Только actuator_relief может обойти current/global hard freeze, токовое
замедление и stall qualification. Сам emergency current trip не обходится
никаким relief. Например, раскрытие middle-ring не разрешает ring joint с
638 mA и ошибкой 11° продолжать нагружаться при global scale=0.
Contact_relief снимает только ограничение, обусловленное геометрией контакта;
оно не доказывает уменьшение нагрузки привода. Human distance rate не даёт
исключений. Operator actuator relief не ждёт yield dwell и проходит быстро,
но не обходит max_direct_step, limits и thumb-index boundary.
Изменение desired с 7 на 10 mm при effective=25 mm ещё требует закрытия, а не
физического открытия: такой запрос не должен обходить контактную границу.

Relief проходит guard без adaptive slowdown, но **не обходит** joint limits,
неисправный сустав, startup blend, max_direct_step и аварийный current trip.
При снятии препятствия без движения оператора шкала постепенно восстанавливается;
разрешённая команда всё равно проходит существующий shaper.

Дополнительные финальные bounds после shaping не дают остаточной скорости
сглаженного режима протолкнуть ограниченный сустав дальше. При clipping
обнуляется скорость только этого сустава (anti-windup); остальные суставы,
таймер и исходная поза ARM blend не переинициализируются.

## Два новых read-only ROS topic

Оба используют `std_msgs/msg/Float64MultiArray`, без нового custom interface.
`layout.dim[0].label` содержит версию и порядок полей; размер и stride равны
длине data, offset=0. Первый элемент — исходный ROS timestamp в секундах.
Если source stamp равен нулю, retargeter ставит своё текущее ROS-время.

`/dg5f/hybrid_contact`, 10 чисел, только в Hybrid:

```text
[stamp_ros_s,
 weight_thumb, weight_index, weight_middle, weight_ring, weight_little,
 distance_thumb_index_m, distance_thumb_middle_m,
 distance_thumb_ring_m, distance_thumb_little_m]
```

Это прежняя чистая `_hybrid_weights()` с прежними параметрами; публикация
происходит после формирования команды. Новые пары не попадают в optimizer.

`/dg5f/safety_proximity`, 8 чисел, во всех режимах ретаргетинга:

```text
[stamp_ros_s,
 thumb-index, thumb-middle, thumb-ring, thumb-little,
 index-middle, middle-ring, ring-little]  # все расстояния в метрах
```

MANO tip indices: `[4, 8, 12, 16, 20]`. Порядок пальцев:
`[thumb, index, middle, ring, little]`; каждый занимает 4 последовательных
сустава в стандартном DG5F joint order.

Bridge принимает только конечные данные нужной длины, веса [0,1], неотрицательные
расстояния и свежий timestamp. Таймаут по исходному времени — 150 ms,
допуск будущего timestamp — 50 ms. Повторный/старый stamp не продлевает свежесть.
ROS clock источника и bridge должны быть согласованы.

NaN/устаревание human topic убирает только соответствующий intent, но не
robot-space защиту thumb-index. Она работает и без обоих human topic.
Недоступная URDF/невалидная robot geometry отключает геометрический компонент;
защита по току, тренду, lead budget и emergency остаётся. Шкала отпускается
плавно. Watchdog здоровья telemetry сохранён.

## Параметры

Файл: `src/lerobot_robot_dg5f/config/bridge.params.yaml`.
Параметры алгоритма читаются при создании узла: после изменения YAML
перезапустите launch. `ros2 param set` не перестраивает объект guard на лету.

В таблице ниже все имена имеют префикс `compliance_`:

| Параметр | По умолчанию | Назначение |
| --- | --- | --- |
| `enabled` | true | Дополнительная compliance; false оставляет threshold guard |
| `joint_slope_soft_ma_s` / `joint_slope_hard_ma_s` | 800 / 3500 | Начало/максимум joint dI/dt риска |
| `total_slope_soft_ma_s` / `total_slope_hard_ma_s` | 1200 / 5000 | То же для распределённой нагрузки |
| `slope_filter_tau_s` | 0.05 | Сглаживание slope |
| `slope_max_reduction` | 0.8 | Максимальное снижение от slope; 0 отключает его scale |
| `contact_preemptive_reduction` | 0.12 | Legacy-параметр, больше не применяется к soft contact |
| `contact_current_start_ma` | 100 | Начало усиления contact-ограничения током |
| `lead_min_deg` / `lead_max_deg` | 1 / 8 | Допустимое дальнейшее опережение feedback |
| `thumb_contact_start_m` / `thumb_contact_full_m` | 0.055 / 0.025 | Зона близости пар с большим пальцем |
| `adjacent_contact_start_m` / `adjacent_contact_full_m` | 0.030 / 0.012 | Зона соседних пальцев |
| `thumb_index_contact_distance_mm` | 25 | Robot-space граница дальнейшего закрытия reference |
| `thumb_index_slowdown_start_mm` | 35 | Начало снижения closing scale thumb-index |
| `soft_attack_tau_s` / `soft_release_tau_s` | 0.03 / 0.06 | Отдельный короткий soft-фильтр, секунды |
| `other_contact_max_reduction` | 0.5 | Максимальное contact-снижение остальных пар; обязательно <1 |
| `yield_enabled` | true | Controlled yield после подтверждения нагрузки |
| `yield_current_ma` / `yield_error_deg` | 240 / 4 | Пороги входа |
| `yield_velocity_deg_s` / `yield_hold_s` | 1 / 0.25 | Максимальная скорость feedback и выдержка |
| `yield_rate_deg_s` | 5 | Максимальная скорость отступления reference |
| `contact_timeout_s` | 0.15 | Свежесть side channel |
| `urdf_path` | `/workspace/models/dg5f/urdf/dg5f_right.urdf` | Геометрия для направления closing |
| `stall_enabled` | true | Дополнительный sustained-stall backstop |
| `stall_current_ma` / `stall_error_deg` | 350 / 8 | Минимальные ток и effective/measured error для stall |
| `stall_velocity_deg_s` / `stall_hold_s` | 0.5 / 0.5 | Максимальная measured speed и непрерывная выдержка |

Имена топиков: `hybrid_contact_topic`, `safety_proximity_topic` в retargeter и
bridge. При переименовании нужно настроить обе стороны.

Ранее настроенные emergency thresholds **не повышены и не откатаны**:

| Защита | soft | hard | trip |
| --- | --- | --- | --- |
| Один сустав, mA | 350 | 650 | 850 |
| Сумма модулей токов, mA | 600 | 850 | 1050 |

`current_guard_trip_hold_s=0.04`, `current_guard_release_tau_s=0.20`.
Trip анализирует абсолютный нефильтрованный ток, а не сглаженный slope.
Краткий всплеск снижает движение; превышение trip в течение 40 ms вызывает
`CURRENT_GUARD_TRIP` и существующий DISARM с reason `OVERCURRENT_GUARD`.
Разгрузочное движение не отменяет этот emergency timer.

Отдельно ток >=350 mA, effective/measured error >=8°, measured speed <=0.5°/s
одновременно на одном суставе в течение 0.5 s без relief вызывают
`STALL_GUARD_TRIP` и DISARM с reason `STALL_GUARD`. Это диагностическая эвристика
устойчивой нагрузки, не доказательство столкновения. Если сработали обе защиты,
приоритет у current trip. Разрыв control ticks >200 ms сбрасывает временные
окна slope/fault dwell: отсутствие данных не считается доказанным sustained
fault; штатный telemetry watchdog обрабатывает потерю связи отдельно.

Freeze не гарантирует снятия уже существующего усилия position controller.
Именно поэтому сохранены emergency DISARM и внутренние защиты Tesollo.

## Примеры

- **Свободное движение:** стабильный малый ток, кончики далеко. Guard передаёт
  desired без изменений; скорость определяется прежним direct shaper.
- **Контакт:** effective=60°, measured=59°, desired=80°, robot thumb-index distance около 25 mm,
  ток соответствующего сустава 450 mA. Contact scale становится 0: команда
  остаётся 60°, а не пытается продавить к 80° и не отскакивает автоматически к 59°.
- **Разгрузка оператором:** при тех же effective/measured desired становится
  20°. Guard пропускает 20° как relief; shaper выдаёт последовательные
  bounded шаги (по умолчанию до 5° за команду), а не мгновенный прыжок.
- **Препятствие убрали:** desired остаётся 80°, ток падает, proximity исчезает.
  Scale восстанавливается постепенно; догоняющее движение ограничено тем же
  max_direct_step. При sustained danger вместо продолжения будет DISARM.

## Что смотреть в dg-status и debug

`/dg5f/lerobot/diagnostics` содержит:

Для всех 7 пар в diagnostics и timeline теперь есть 7-элементные массивы:
`robot_measured_pair_distances_mm`, `robot_effective_pair_distances_mm`,
`robot_desired_pair_distances_mm`, `pair_tracking_scale`. Порядок прежний:
thumb-index/middle/ring/little, index-middle, middle-ring, ring-little.
В CSV каждый элемент имеет суффикс имени пары. Pair scale — минимум реально
разрешённого tracking gain участвующих в движении суставов, включая current
limiting; это не только raw contact weight.

Для yield добавлены `yield_active`, `yield_joints` (индексы 0…19),
`yield_delta_deg` (20 signed шагов в градусах). При yield joint_tracking_scale=0:
это не продвижение к нагружающему desired, а контролируемое отступление.

Прежние пять thumb-index полей сохранены:
`thumb_index_human_contact_weight`, `thumb_index_measured_tip_distance_mm`,
`thumb_index_effective_tip_distance_mm`, `thumb_index_desired_tip_distance_mm`,
`thumb_index_contact_tracking_scale`, а также `robot_contact_limit_active`.
Те же поля включены в timeline.csv. События `ROBOT_CONTACT_LIMIT_ACTIVE` и
`ROBOT_CONTACT_LIMIT_RELEASED` теперь ведутся отдельно для каждой пары:
только `pair`, её 3 расстояния и scale, без всей покадровой телеметрии.
Прежний debounce переходов 200 ms сохранён; счётчики включены в summary.txt.
Расстояния — по feedback/reference ДО
текущего шага, шкала — для текущего решения. `compliance_age_ms` показывает
возраст решения, если output остановлен.

Остальные ранее существовавшие поля:

- 20-элементные `joint_current_scale`, `joint_current_slope_ma_s`,
  `joint_slope_scale`, `joint_contact_scale`, `joint_tracking_scale`, `joint_lead_budget_deg`;
- global current/slope scales, `total_current_ma`, `total_current_slope_ma_s`;
- `hybrid_contact_weights`, `pair_distances_m`, `pair_distance_rates_m_s`, `pair_contact_weights`;
- `contact_signal_valid`, freshness/age обоих источников, `compliance_age_ms`, FK error;
- `limited_fingers`, `contact_limited_pairs`, прежний `current_guard_limited_joints`;
- `compliance_active`, `current_guard_min_scale`, `stall_duration_s`.

Теперь `contact_signal_valid` означает валидную robot geometry;
`pair_contact_weights` — robot proximity с human intent-gate для thumb-middle/ring/little;
adjacent weight без load является лишь диагностикой. `pair_distances_m` и
`pair_distance_rates_m_s` по-прежнему относятся к
человеку, как и исходные ROS topic; они не переименованы и не выданы за feedback.

`joint_tracking_scale` — реально разрешённый gain guard, включая exhausted
lead budget; скорость финального shaper может быть ниже из-за startup blend.
Ориентируйтесь также на свежесть снимка, effective и measured command.

`timeline.csv`: шкалы/производные по индексам суставов, расстояния/веса по
именам пар; прежние target/effective/measured/current остаются.
`events.jsonl`: `COMPLIANCE_ACTIVE`, `COMPLIANCE_RELEASED`,
`CURRENT_GUARD_TRIP`, `STALL_GUARD_TRIP`. ACTIVE немедленный, RELEASED после
200 ms непрерывно неограниченного движения; это debounce **только событий**.
Каждокадровые переходы доступны в timeline. DISARM отдельно завершает эпизод.
NaN/Inf в JSON превращаются в null. `summary.txt` содержит минимальный scale,
число ограниченных samples по парам и число compliance/trip событий.
Manifest содержит порядок полей, единицы и конфигурацию guard.

## Проверки без физической DG5F

Доработка после физического теста проверяется только пятью тестами в
`test_load_response.py`: сохранение TI boundary, запрет geometry relief bypass,
gradual yield, свободные adjacent pairs при малом токе, мягкий thumb-ring без
необоснованного нуля. Последний тест также проверяет все pair-поля CSV и
независимые event-переходы пар. Полный suite для этой доработки не нужен.

Для robot-space доработки достаточно целевых `test_robot_contact.py` (4 сценария)
и существующих `test_current_guard.py`. Полный набор ниже — опциональный,
для этой доработки повторно не запускался. Четыре сценария: human pinch при
далёком роботе; near-contact + растущий ток/граница reference; быстрый relief
с событиями/debug CSV; сохранение emergency trip. Используется только fake/mock.

```bash
cd /home/yoba/Documents/work/hand_hybryd_retargeting
bash scripts/stack.sh compile
bash scripts/stack.sh test
bash scripts/stack.sh smoke
bash scripts/stack.sh dataset-mock --repo-id local/compliance-check-01 --with-camera --interrupt-active
```

Для повторного dataset-mock используйте новое имя repo-id, чтобы не перезаписать
предыдущий результат. Smoke использует явный mock backend, изолированный ROS
domain, синтетический TCP-клиент и headless MuJoCo; проверяет оба contact topic,
fault map, grace hold и DISARM по истечении 15 s. Unit-тесты охватывают все
12 сценариев из задания, FK сравнён с finite differences и Pinocchio.

При следующем ручном тесте уже запущенного pipeline:

```bash
bash scripts/stack.sh dg-status
bash scripts/stack.sh debug-record
```

Сначала проверьте свободное движение, затем обычное сведение и разведение
кончиков без принудительного удерживания пальцев. Не создавайте аварийную
нагрузку специально. При неожиданном движении остановите output:

```bash
bash scripts/stack.sh disarm
```

Этот гайд не выполняет ARM и не запускает аппаратный backend. Решение о
готовности реальной кисти и ручном ARM остаётся за оператором.

## Ограничения

Robot fingertip distance + URDF производная — лёгкая подсказка о направлении
контакта, а не collision checker всех meshes. Human proximity — только intent.
Граница 25 mm относится к расстоянию URDF tip links, а не зазору между поверхностями.
Этот patch не проверяет физически, что 25 mm оптимальны для данного экземпляра.
Не видит надёжно контакты с
объектами, фалангами и все возможные пары пальцев; их нагрузка отслеживается
по току. Погрешность геометрии и масштаба человеческой/роботизированной кисти
сохраняется. Ток не измеряет силу контакта напрямую. Неизменёнными оставлены
backend loop 200 Hz, recovery, ARM/grace, startup blend, rj_dg_5_1, транспорт
Quest, оптимизаторы, MuJoCo и LeRobotDataset recorder.

## Файлы этой доработки

Новые:

- `src/dg5f_teleop/dg5f_teleop/contact_signals.py` — wire contract и 7 расстояний;
- `src/lerobot_robot_dg5f/lerobot_robot_dg5f/contact_kinematics.py` — read-only FK/Jacobian;
- `src/dg5f_teleop/test/test_contact_signals.py`;
- `src/lerobot_robot_dg5f/test/test_compliance.py`;
- `src/lerobot_robot_dg5f/test/test_compliance_bridge.py`;
- `src/lerobot_robot_dg5f/test/test_robot_contact.py` — 4 целевых robot-space сценария;
- `src/lerobot_robot_dg5f/test/test_load_response.py` — 5 проверок по физическому отчёту;
- `docs/ADAPTIVE_COMPLIANCE.md` — этот гайд.

Изменены:

- `src/dg5f_teleop/dg5f_teleop/retarget_node.py` — только contact-публикация;
- `src/lerobot_robot_dg5f/lerobot_robot_dg5f/current_guard.py` — adaptive tracking;
- `src/lerobot_robot_dg5f/lerobot_robot_dg5f/command_shaper.py` — финальные bounds;
- `src/lerobot_robot_dg5f/lerobot_robot_dg5f/dg5f.py` — передача bounds в shaper;
- `src/lerobot_robot_dg5f/lerobot_robot_dg5f/ros_bridge_node.py` — подписки, guard, diagnostics/events;
- `src/lerobot_robot_dg5f/lerobot_robot_dg5f/debug_recorder.py`;
- `src/lerobot_robot_dg5f/lerobot_robot_dg5f/debug_recording.py`;
- `src/lerobot_robot_dg5f/config/bridge.params.yaml` — новые параметры, прежние thresholds сохранены;
- `src/lerobot_robot_dg5f/package.xml` — зависимость от общего contact contract;
- `src/dg5f_teleop/test/test_velocity_profile.py` — отсутствие влияния observer на output;
- `src/lerobot_robot_dg5f/test/test_debug_recording.py`;
- `src/lerobot_robot_dg5f/test/test_recovery.py` — mock bridge/effective output/emergency проверки;
- `scripts/unity_smoke_inside.sh`, `scripts/validate_smoke.py` — новые topic и прежний grace;
- `README.md` — место guard в цепочке и ссылка на гайд.

Изменения dataset recorder и current-guard patches, уже находившиеся в рабочем
дереве перед этой задачей, сохранены; они не откатывались и не выдаются за новую
реализацию compliance. `main` не изменялась.
