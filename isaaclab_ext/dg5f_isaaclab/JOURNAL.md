# Журнал работы — dg5f_isaaclab

Выводы и причины, новые записи сверху, существующие записи не переписываются.
История до этой ветки: ../../JOURNAL.md. Локальный журнал создан, поскольку ночное
задание разрешает изменения только внутри isaaclab_ext/dg5f_isaaclab.

## 2026-10-02 — H1: смок на 2 сидах был слеп, на 8 сидах эффект есть, но не походочный

`scripts/gait_multiseed.py`, отчёт logs/gait_research/H1_MULTI_SEED_REPORT.md,
данные logs/gait_research/h1_multiseed.json. 24 оценки: 3 чекпойнта x 8 сидов x 128
эпизодов, детерминированные средние действия, стабильные сбросы для ВСЕХ ветвей
(включая B, иначе старт из готовой середины перехода зачлось бы как навык). Пороги
походки заморожены и побитово одинаковы во всех 24 прогонах: meaningful_displacement_m
= 0.0109 м при кубе 60 мм, release_min_steps = 4.

Почему ночной вердикт `not_supported_by_400_iteration_smoke` оказался несостоятелен:
разброс B по сидам на `meaningful_events_per_episode` = 0.914..1.484, а две оценки
смока дали ровно 0.914 и 1.484. Смок попал в оба хвоста одного распределения и
прочитал это как противоречие между сидами. Диапазон A (1.047..1.344) целиком внутри
диапазона B.

Парно внутри сида (B-A, 95% ДИ бутстрапом по 8 сидам, не p-значения):
- награда +25.6 [15.7, 35.7], 8/8 сидов
- опора на два кончика +0.272 с [0.143, 0.414], 8/8
- drop -0.059 [-0.088, -0.030], 7/8 в пользу B
- целей/эп +0.662 [0.171, 1.099], 6/8
- осмысл.события/эп +0.129 [0.010, 0.251], 6/8
- доля успешных циклов +0.011 [-0.017, 0.039], 4/4 -- плоско
- разных контактных масок +0.009 [-0.325, 0.387], 3/5 -- плоско

Вывод: смешанные сбросы дали лучшую политику в целом, но НЕ более шаговую. Две
метрики, ради которых ставился H1 (частота успешных циклов перехвата, разнообразие
масок), не двинулись вообще. По правилам §8 задания это AMBIGUOUS, а не POSITIVE.

Контроль, который делает результат нетривиальным: A и B получили ОДИНАКОВЫЕ 400
итераций от общего чекпойнта. A за них не улучшилась (drop 0.662 против 0.641 у
BASELINE@3600, held 0.890 против 0.883) -- то есть прирост B это эффект вмешательства,
а не лишних итераций.

§9 (не является ли компетентность B состояние-зависимой) -- ответ отрицательный, и это
важнее самого вопроса. Избыточная дисперсия у B реальна: sd средних по сидам 0.173
против пуассоновского предсказания sqrt(mean/128) = 0.101, отношение 1.72; у A
отношение 1.06, то есть чистый шум выборки. Но распределение по эпизодам у A и B
почти совпадает по форме, и доля эпизодов БЕЗ осмысленных событий идентична:
0.417 у A против 0.419 у B (0.417 у BASELINE). Весь прирост B сидит в верхнем хвосте:
эпизодов с 5 событиями 12 -> 28, с 6 событиями 10 -> 18, с 7 событиями 2 -> 6.
Значит вмешательство не научило B пробовать перехват там, где A не пробует; оно
заставило уже перехватывающие эпизоды перехватывать больше. Отсюда и тяжёлый хвост,
который качает среднее по сиду, -- и ровно поэтому двух сидов было принципиально
недостаточно.

Ловушка окружения: структура eval-JSON вложенная, метрики лежат под ключом
`<tag>/model_<N>`, а не в корне; в gait-секции нет `contact_switch_distance_p99`
(есть только mean/p50/p90) и ключ масок называется `distinct_masks_per_episode`.
Имена проверены по реальному JSON до запуска -- иначе таблица собралась бы пустой
без всякой ошибки.

Не доделано: LONG_H1 (продолжение ОБЕИХ ветвей от @400 на ~2000 итераций, оценки
каждые 500 на панели из 3 сидов) не запускался -- жду решения пользователя. H2
(контактный гейтинг награды) не начат. Коммитов по этой кампании нет.

## Ночное сглаживание — 2026-10-01T21:53:14.068053+00:00

Отчёт: logs/night_smoothing/FINAL_REPORT.md. Ветка: H; завершение: drop_materially_worse_for_two_evaluations. Физика и delta1° неизменны. Коммитов нет.

## 2026-10-01 — сохранение результатов в Git по запросу пользователя

В reports/snapshot_2026-10-01 сохранены пять ключевых checkpoint (E2/3600,
std0.5, A/B+100, 10°+исследование), конфиги и JSON-результаты. manifest.json
содержит исходные пути, размеры и SHA256; копии проверены по хешам. Всего
50 файлов, около5 МБ. Полные логи/промежуточные модели остаются в logs.
Вместе со снимком коммитятся код, тесты, отчёты и локальный журнал кампании.

## 2026-10-01 — стартовые eval и 100 итераций A/B: восстановление проверено

По запросу пользователя после аудита выполнены два initial deterministic eval
и независимые продолжения по100 итераций. A: std0.5/entropy0.001, B: исходный
E2 std5.12175/entropy0.005. Начальные eval совпали по всем метрикам кроме std:
held0.859375, drop0.6171875, целей9.703125, reward139.09709.

Новый opt-in --verify_resume_state в локальном train.py проверяет до обучения
все модельные тензоры/Adam, next_iteration3600, LR0.0002562890625 и неизменность
полного env.yaml кроме log_dir, включая reward. Обе проверки прошли. Финальные
model_3699: у каждого параметра Adam +2000 шагов, исходные SHA256 не изменены.

A после100: std0.502193, held0.859375, drop0.6328125, целей9.3828125,
reward136.74; B: std5.244721, held0.84375, drop0.640625, целей9.0859375,
reward130.12. A сохранил навык, но снижения падений пока нет; разница drop
между A/B — один эпизод из128, победителя по одному smoke не объявляем.
LR адаптивно разошёлся до3.375e-5 уA и5.7665e-4 уB; начальный LR был одинаков.
Насыщение A осталось практически прежним (near-limit0.86934→0.86882).

Отчёт reports/e2_std_ab_smoke.md; состояние logs/e2_std_ab_smoke/state.json.
Драйвер scripts/e2_std_ab_smoke.py.16 CPU-тестов прошли. Оба прогона и оценки
завершены, новых обучений нет, reward/physics/control не менялись.

## 2026-10-01 — аудит E2/3600: редкие превышения FD остаются после сброса; std-only подготовлен

Дважды выполнен velocity_audit.py на model_3599: Stream/A/20°, 128 сред, seed1234,
1440 управляющих шагов, выборка на каждом физическом шаге, исключение разностей
через reset. E2 FD max6.1077, худший per-joint p99 3.1299 рад/с; доля >π по всем
20 суставам0.04222%. Нулевая политика FD max6.9918 возникает сразу после reset.
Добавлен разбор возраста эпизода: после исключения первой секунды E2 FD max3.7433,
>π0.03977% по19 активным суставам; всего3 отсчёта >1.1π из6431842. У нулевой
после1с max1.0270 и превышенийнет. Не доказано, что E2 эксплуатирует эти события,
но строгая граница π не соблюдается. Физика/награда не менялись; обучение не запущено.

make_warm_start.py --reset_std_only сохраняет всё кроме std/log_std, включая
критика, нормализаторы, Adam(также моменты std), LR и iter. Реальная копия:
logs/rsl_rl/dg5f_cube_direct/2026-10-01_e2_std05_preserved/model_3599.pt.
Побитовое рекурсивное сравнение: изменён только std, до0.5; iter3599, AdamLR
0.0002562890625. Entropy0.001 задаётся при будущем запуске, не в checkpoint.
14 CPU-тестов и py_compile прошли. Подробности/оговорки: reports/e2_velocity_audit.md;
сырые данные logs/e2_velocity_audit/audit{,_detailed}.json.

## 2026-10-01 — итог 10° + исследование: цели появились, падения остаются ограничением

Кампания завершена. Детерминированно по 128 эпизодам: 10°/std1/entropy0.005
после 600 итераций даёт held 0.765625, drop 0.671875, целей/эп 2.6328125,
max целей подряд 22, min ошибка 2.200°, std 1.7264. Прежний E5 (тот же бюджет
600 и 10°, std0.2/entropy0) давал held 0.09375, drop 0.6171875, целей/эп
0.1015625. Прирост held +67.2 п.п., целей/эп примерно в 25.9 раза; drop вырос
на 5.5 п.п. Следовательно, прежний отрицательный E5 не опровергал 10° с
исследованием. Это измерение пакета std+entropy, не отдельной причины.

E2/20° после 3600: held 0.859375, drop 0.6171875, целей/эп 9.703125,
max подряд 42, min ошибка 2.187°, std 5.1217. Оба новых результата проходят
пороги held и целей/эп, но не drop<=0.15. Нельзя сравнивать число целей на 10°
и 20° как одинаковую сложность. Длинное продолжение E2 и отдельный опыт 10°
завершены; новых обучений в очереди нет. Источник: logs/e2_continuation/state.json;
таблица всех контрольных точек: reports/e2_continuation.md.

## 2026-10-01 — продолжение E2 и 10° с исследованием завершены

Измеренные оценки всех checkpoint: reports/e2_continuation.md и logs/e2_continuation/state.json. E2 получил полные 3600 итераций, отдельный 10°-опыт — 600 от warm_start_std1. Параметры по ходу не подбирались.

## 2026-10-01 — GUI просмотра E2 падал до загрузки checkpoint из-за занятой GPU

Два пользовательских запуска play.py с model_3599.pt подтверждены commandLine в
kit_20261001_181941.log и kit_20261001_182007.log. В обоих Vulkan падает при запуске
GUI: Out of GPU memory allocating resource, ERROR_OUT_OF_DEVICE_MEMORY, затем
ERROR_DEVICE_LOST. Во втором логе первые ошибки на строках 3516/3521. Это не ошибка
загрузки E2: параллельное обучение 10° продолжает занимать 4362 МиБ.

В локальный play.py добавлена проверка до создания AppLauncher/Kit: GUI получает
общую блокировку с драйвером обучения/оценки и проверяет compute-процессы GPU.
Без --wait_for_gpu команда завершается понятной ошибкой, с флагом ждёт окончания
ВСЕЙ кампании, включая оценку, чтобы не стартовать в промежутке между процессами.
Ничьи процессы не останавливаются. Три CPU-теста guard прошли; GUI параллельно
обучению повторно не запускался. Параметры обучения и политика не менялись.

## 2026-10-01 — E2 продолжен: бюджет отбора не является бюджетом сходимости

После разбора кривых вывод «исследование ухудшает удержание» как общий вывод о
процессе обучения снимается. Конечный deterministic drop 0.641 против 0.078 у
исходной политики — корректное сравнение одного среза, но не траектория обучения.
Средние по сотням итераций E2 на позднем участке: training reward −97.52 → −83.63
→ −67.34 → −58.43 → −45.99, training drop 0.890 → 0.875 → 0.829 → 0.828 → 0.790.
Это аргумент продолжить. Строгой монотонности с самого начала нет, а последняя
точка drop 0.579 заметно оптимистичнее среднего последней сотни. Графики/сырые
строки: reports/e2_trend_interpretation.md и exploration_training_curves.json.
E2 завершал максимум 5 целей в эпизоде. E3 тоже имеет held >0.28 (0.289), поэтому
исследование нельзя объявлять единственным действующим фактором.

Запущено продолжение E2 с model_599.pt, ещё 3000 итераций до суммарных 3600.
Параметры E2 сохранены: 20°, A, 1024 среды, 32 шага, gamma 0.99, entropy 0.005;
std/актор/критик/нормализаторы/Adam загружены из checkpoint. Новый локальный флаг
--restore_continuation_state восстанавливает PPO.learning_rate из Adam:
0.00011390625. Без этого RSL-RL 3.1.2 перезаписал бы загруженный LR значением
конструктора 0.001 при первом adaptive update. Исправлен только локальный wrapper,
RSL-RL не менялся. Номер следующей итерации 600, а не повторный 599.
Подтверждение: logs/e2_continuation/train_from_600.log, Loading model checkpoint
from и [CONTINUE]. Физическое состояние среды и RNG не сохранялись исходным
checkpoint; при первом resume среды сбрасываются, далее обучение непрерывное.

E2 не использует train_stream, поэтому ни watchdog 0.35, ни stage_max_iterations
2000 его не остановят. После непрерывного обучения драйвер оценит сохранённые
промежуточные checkpoint и конечный model_3599.pt, последовательно на одной GPU.
Затем отдельный опыт 10° + исследование: 600 итераций от warm_start_std1,
entropy 0.005, gamma 0.99, train_stream --bootstrap. Для него введён отдельный
порог --action_std_watchdog 3.0; низкошумовой default остаётся 0.35 для прежних
запусков. 3.0 — оперативный порог этого режима, не измеренная граница полезности
исследования. Пороги гейта не менялись. Gamma 0.998 в новом плане отсутствует.

Точка продолжения: scripts/continue_e2.py, logs/e2_continuation/state.json и
driver.log; отчёт автоматически обновляется в reports/e2_continuation.md.
Тест восстановления LR/итерации и исходные проверки управления: 73 passed,
8 subtests passed. Продолжение не сбрасывает std и не создаёт новый warm start.

## 2026-10-01 — итог завершён: улучшение достижения целей без надёжного хвата; порт и эталон проверены

Все шесть E* завершили по 600 итераций от правильных warm start, итоговые checkpoint
model_599.pt. Детерминированно E2: held 0.4609375, drop 0.640625, целей/эпизод
0.6328125, min ошибка 4.02°. Это +42.2 п.п. held и +56.3 п.п. drop к исходной
политике: исследование двигает достижение цели, но ухудшает удержание. E3: held
0.289, drop 0.508, целей/эп 0.391; у него вдвое больше переходов при том же числе
итераций, поэтому эффект батча не отделён от бюджета опыта. E6: held 0.297,
drop 0.398, целей/эп 0.297; относительно E2 меньше падений, но меньше завершённых
целей. E1 (held 0) и E5 (held 0.094 против 0.250 на 10°) дали отрицательный
результат. E4: held 0.078, drop 0.555. Ни один гейт не прошёл. Полная таблица,
включая награду/std и точные источники опор: reports/overnight_ab_repose.md.

Порт DG5F валиден в runtime: 19 действий / 146 наблюдений, 600 шагов без NaN,
куб 60 мм / 50 г, проверено совпадение reward/reset/tolerance с Shadow.
25 PPO-итераций при 1024 средах прошли: пик compute 4484 МиБ, всей карты 5331 МиБ.
20 итераций при 2048 тоже прошли: 4880 / 5727 МиБ. 2304 и выше упали с настоящим
OOM (logs/overnight_repose/repose_smoke_2304.log: CUDA out of memory;
4096 — PhysX allocation failures). Измеренная рабочая точка — 2048, шаг поиска
256. Это максимум короткого смока в текущем окружении, не гарантия долгого прогона.

Предыдущая запись «Порт: failed» отражала агрегированный статус сбоившего ЭТАЛОНА,
а не провал смока DG5F. Опубликованный Shadow checkpoint имеет старые отдельные
нормализаторы; RSL-RL 3.1.2 ожидает их внутри model_state_dict. Исправлена только
совместимость локального play.py: перенос mean/var/std без изменения, count=0
как неиспользуемая inference-заглушка; исходный checkpoint сохранён, RSL-RL не
правился. Повторный эталон при 32 средах прошёл 1200 шагов, consecutive_successes
8.3243. Лог: logs/overnight_repose/shadow_reference.log; старый сбой сохранён как
shadow_reference_legacy_failure.log. Копию *_rsl3_inference.pt нельзя использовать
для возобновления обучения: исходное число наблюдений нормализаторов неизвестно.

Итоговые проверки: 76 passed, 8 subtests passed (logs/overnight_ab/tests_all.log).
GPU свободна, новых длинных прогонов не запущено. Длительное обучение порта не
входило в эту ночь: смок доказывает исполнение/VRAM, но не обучаемость. Контроллер
порта штатный absolute; delta-интегратор и задержку stream этот тест не проверяет.
Работа после ec922e8 оставлена незакоммиченной в experiment/dg5f-overnight-ab-repose.

## 2026-10-01 — ночная лестница — измеренный итог

Завершено 6/6 A/B, прошли гейт: никто. Порт: failed. Числа, сдвиги и ограничения: reports/overnight_ab_repose.md; машинные результаты — logs/overnight_ab/results.json и logs/overnight_repose/results.json.

Вывод нельзя делать по training drop_rate: таблица построена по отдельным детерминированным оценкам. E5 сравнивается с 10°, E4 — с правильной формой награды. Короткий смок порта не отделяет нехватку бюджета от дефекта управления.

## 2026-09-30 — ограничения диагностического порта и фактическая форма E4

Предположение «DG5F action_space=20» не подтверждается исходниками: URDF содержит
20 суставов, но rj_dg_5_1 отключён, активных действий 19. У upstream full observation
2*num_dofs + 24 + 13*num_tips + num_actions. Для 19 подвижных осей и 5 кончиков это
146 (не Shadow 157). Его unscale делит на upper-lower: нашу блокировку лимитами
[0,0] напрямую перенести нельзя, иначе наблюдения NaN. Отдельный ассет порта
представляет уже неподвижную ось fixed при 0, не сливая тела. CPU-тест проверяет
тождественность всех остальных joint-элементов, тел, масс и геометрии. Исходный
URDF и stream не меняются. Runtime-проверка и замер VRAM ждут освобождения GPU.

Это порт штатного ABSOLUTE-контроллера upstream, а не испытание нашего интегратора
и задержки. Поэтому положительный результат подтвердит работоспособность кисти
в этом варианте управления; отрицательный короткий смок не докажет дефект кисти
или управления. Эту оговорку нельзя убрать из итогового вывода.

E4 переключает существующую orientation_state_reward: exp(-(err/sigma)^2), sigma=10°,
а не exp(-err/sigma) из описания ночного задания. Сохранён именно запрошенный
флаг env.orientation_baseline=false, никаких дополнительных reward-правок нет.

## 2026-09-30 — ночная лестница запущена от проверенного warm start

Исходное состояние сохранено коммитом ec922e8 по прямому запросу пользователя;
работа ведётся в experiment/dg5f-overnight-ab-repose. Последующие коммиты не нужны.
E2 действительно загрузил 2026-09-29_00-00-01_warm_start_std1/model_0.pt:
logs/overnight_ab/E2_exploration/train.log содержит Loading model checkpoint from.
CPU-проверка сравнила все тензоры: после штатного make_warm_start.py отличается
только std (0.2 -> 1.0), критик и нормализаторы совпадают побитово.

Лестница фиксирована: E2/E1/E5/E4/E3/E6, по 600 итераций, 1024 среды; оценки по
128 эпизодам с seed 1234. E5 идёт через train_stream.py одним сегментом на 600
итераций: этим сохраняется 10°/A весь прогон, пороги гейта не меняются. Для E4
оценщик явно получает orientation_baseline=false, иначе его episode_reward
измерял бы другую награду. Успех/падение оцениваются одинаково с другими E*.

Точка продолжения: logs/overnight_ab/results.json и driver.log. Драйвер пропускает
завершённые оценки, перепускает неудавшиеся прогоны от исходного warm start;
ошибка загрузки warm start останавливает всю лестницу. GPU-процессы очищаются
только в собственной process group, чужие процессы не трогаются.


## 2026-10-04 — controlled gait decision tree started

User requested up to14h autonomous decision tree, no commits. ROOT verified as
`2026-10-02_18-15-45_gait_H1_B/model_3999.pt`, SHA256
`48522e7760cdf3d9f7913930ddb92a5f5645e545842b0168563d0177c092d5cb`.
CONTROL_LONGISH starts at iteration4000 with exact model/Adam and std5.550039,
LR0.0002562890625000001, entropy0.005. Physics, goals20deg and mixed70/30 resets
remain fixed. H2 gates only task rewards/eligibility; conditional H3/H4/H5 add
observations via zero columns and preserved optimizer state. Actual RSL actor and
critic outputs match exactly on CPU probes for all three expansions. Frozen decisions,
process/deadline state and reproducible dump deletion manifest: `logs/gait_tree/`.
Results are pending; starting a run does not establish a gait improvement.

### 2026-10-04 — smoke results and explicit selection review

CONTROL, H2, H3, H4, H4+H2 and H5 completed300 updates each; all resume checks passed.
H4 and H5 were replicated on8 evaluation seeds, with H5 compared against contact flags.
H4 vs matched CONTROL: cycles0.3320->0.3838 (+15.59%,6/8 seeds), recovery+0.00486
(7/8), goals10.789->11.230, drop0.686->0.631. The initial automatic selector rejected
H4 because an assistant-added20% cycle-gain threshold was not reached. This cutoff was
stronger than the user's qualitative replication requirement. On review it was explicitly
superseded for an exploratory H4-only continuation; the original failure under that rule
remains documented. No metric threshold was lowered and no physical parameter changed.

H4's gain partly reflects longer survival: time-normalized cycles improve8.5%,6/8 seeds,
but the paired bootstrap interval includes0. This is a modest signal, not established
organized gaiting. H2/H4+H2 reduce gait; contact flags alone do not improve it; H5 improves
some gait metrics but increases drop. Full measurements and selection reasoning are in
logs/gait_tree/SMOKE_REPORT.md and SELECTION_REVIEW.json.

Long H4 runs in tmux gait_tree_h4 under the original14h deadline, from
2026-10-04_06-16-57_gait_tree_H4_GRASP_MECHANICS/model_4299.pt. First continuation
loaded actor, critic, normalizers and Adam exactly; iteration4300, std5.85, fixed LR.
Final reports/videos and a matched post-FIFO action audit are queued; final review pending.


## 2026-10-04 — Gait decision tree completed

H4 grasp-mechanics observation was the single long candidate after the documented selection review. Completed3000 additional updates; selected model7299. Full8-seed stable-reset evaluation (1024episodes): strictcycles/episode0.55664, meaningful relocations1.89160, goals13.94434, drop0.59277. Short H4:0.38379cycles,11.23047goals,0.63086drop. Long-minus-short cycles positive8/8seeds; pairedmean+0.17285, bootstrap95%[0.11719,0.21777]. Cycles/second0.02266→0.03273; this gain is not just longer survival. Causal H4 attribution remains limited by no equally trained long control and only one training seed.

Verdict PARTIAL: more rare strict gait cycles, no established organized repeated gait. Dominant bottleneck not established; mechanics awareness is the strongest tested candidate. H2 and H4+H2 harmed useful gait; flags alone and timers did not qualify under the documented selection tests. No physics, action, threshold, or cache changes. Final std9.10093; actual delivered near-limit fraction0.94558 on matched seed31415; finite-difference qdot p993.03486rad/s, maximum3.53509 (no claim all samples obeypi).

Training/evaluation/video/delivery-audit queue finished17:34MSK within original budget. Final manual review on later status turn: sampled video sequences and both plots checked;11 tests passed; fixed input hashes and originalHEAD3d9accc7f2e180ee04a1054466a07df52edf3ae2 unchanged. No commits. All four requested3-episode videos and exact demo command saved underlogs/gait_tree. FINAL_REPORT.md and final_results.json contain full tables, paired comparisons, contact/finger diagnostics, learning curve, limitations and next experiment: second training seed plus matched long control.
