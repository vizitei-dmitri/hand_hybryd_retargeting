# Проверка старта и короткое продолжение E2 A/B

Статус: completed

A: std=0.5, entropy=0.001. B: исходный std≈5.12, entropy=0.005. По 100 итераций, 1024 среды, rollout32, seed42, gamma0.99, A/20°. Оценки: mean action, 128 эпизодов, seed1234. Это сравнение пакета std+entropy, не изолированный эффект std.

Перед первым обновлением сверяются все тензоры модели, Adam, LR, номер итерации и вся конфигурация среды (кроме пути логов), включая награду. Состояние симулятора/RNG исходный checkpoint не сохранял.

| Вариант | held | drop | целей/эп | reward | std | near limit |
|---|---:|---:|---:|---:|---:|---:|
| A_before | 0.8594 | 0.6172 | 9.7031 | 139.097 | 0.5000 | 0.8693 |
| B_before | 0.8594 | 0.6172 | 9.7031 | 139.097 | 5.1217 | 0.8693 |
| A_after100 | 0.8594 | 0.6328 | 9.3828 | 136.737 | 0.5022 | 0.8688 |
| B_after100 | 0.8438 | 0.6406 | 9.0859 | 130.122 | 5.2447 | 0.8767 |

A: `/home/yoba/Documents/work/hand_hybryd_retargeting/isaaclab_ext/dg5f_isaaclab/logs/rsl_rl/dg5f_cube_direct/2026-10-01_20-08-06_e2_std_ab_A_100/model_3699.pt`

```json
{
  "checkpoint": "/home/yoba/Documents/work/hand_hybryd_retargeting/isaaclab_ext/dg5f_isaaclab/logs/rsl_rl/dg5f_cube_direct/2026-10-01_e2_std05_preserved/model_3599.pt",
  "checkpoint_sha256": "9accb125331c542a4151e01f897ad6ded729bc0b11b4e7e21ce983e8a4397ff3",
  "all_model_tensors_exact": true,
  "critic_preserved": true,
  "normalizers_preserved": true,
  "optimizer_exact": true,
  "environment_and_reward_exact_except_log_dir": true,
  "source_iteration": 3599,
  "next_iteration": 3600,
  "ppo_learning_rate": 0.0002562890625000001,
  "Adam_lrs": [
    0.0002562890625000001
  ],
  "entropy_coef": 0.001,
  "std_mean": 0.5,
  "std_min": 0.5,
  "std_max": 0.5,
  "std_per_action": [
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5
  ],
  "model_tensor_count": 21,
  "Adam_state_entries": 13
}
```

B: `/home/yoba/Documents/work/hand_hybryd_retargeting/isaaclab_ext/dg5f_isaaclab/logs/rsl_rl/dg5f_cube_direct/2026-10-01_20-17-13_e2_std_ab_B_100/model_3699.pt`

```json
{
  "checkpoint": "/home/yoba/Documents/work/hand_hybryd_retargeting/isaaclab_ext/dg5f_isaaclab/logs/rsl_rl/dg5f_cube_direct/2026-10-01_14-39-27_e2_continue_3600/model_3599.pt",
  "checkpoint_sha256": "ee1f618f08abb34c9c23a09fbd386cc432a61022b93c838f8349554d89bd3c22",
  "all_model_tensors_exact": true,
  "critic_preserved": true,
  "normalizers_preserved": true,
  "optimizer_exact": true,
  "environment_and_reward_exact_except_log_dir": true,
  "source_iteration": 3599,
  "next_iteration": 3600,
  "ppo_learning_rate": 0.0002562890625000001,
  "Adam_lrs": [
    0.0002562890625000001
  ],
  "entropy_coef": 0.005,
  "std_mean": 5.121748924255371,
  "std_min": 3.4325952529907227,
  "std_max": 5.9443793296813965,
  "std_per_action": [
    4.503330707550049,
    4.657010078430176,
    3.4325952529907227,
    3.975393772125244,
    5.724458694458008,
    5.245089530944824,
    5.2438178062438965,
    5.098825931549072,
    5.447271347045898,
    4.9137749671936035,
    5.344310283660889,
    5.683235168457031,
    5.509978294372559,
    5.231533050537109,
    5.6789326667785645,
    5.332801818847656,
    5.9443793296813965,
    5.150691032409668,
    5.195809364318848
  ],
  "model_tensor_count": 21,
  "Adam_state_entries": 13
}
```

## Проверенные итоги

Стартовые eval A/B совпали по всем сохранённым метрикам, кроме action_std:
A=0.5, B=5.1217489. Для каждого запуска использовался отдельный процесс
симулятора, 128 эпизодов, seed1234. Побитовое сравнение checkpoint до обучения
подтвердило совпадение всех полей кроме std.

Оба старта прошли --verify_resume_state до первого обновления: 21 тензор
модели, 13 записей состояния Adam, критик и нормализаторы совпадают с источником.
Начальная итерация3600, восстановленный PPO/Adam LR0.0002562890625 у обоих.
Вся конфигурация среды, включая reward/physics/control, совпала с исходной
(исключён только log_dir). Дополнительный diff agent.yaml подтвердил отсутствие
неплановых изменений PPO: logs/e2_std_ab_smoke/{A,B}_agent_config_diff.json.
Награда не хранится в checkpoint: совпадение проверено по конфигурации среды
и одинаковому стартовому deterministic return139.0971.

Каждый вариант прошёл ровно100 итераций до model_3699.pt. По6 тензоров актора
и критика обновлены; у каждого параметра Adam добавлено2000 шагов
(100 итераций ×5 эпох ×4 минибатча). Исходные SHA256 сохранились.
Проверки: final_checkpoint_verification.json рядом с состоянием кампании.

Финальный std: A0.502193, B5.244721. Адаптивный LR: A0.00003375,
B0.000576650390625. Это измеренные конечные значения после обновлений;
восстановление начального LR проверено отдельно и у обоих было одинаковым.
Поэтому нельзя трактовать последующую динамику как эксперимент с фиксированным LR.

A сохранил выполнение целей (held0.8594, целей/эп9.3828), но drop0.6328
не улучшился относительно стартовых0.6172. B: held0.84375, целей/эп9.08594,
drop0.640625. Отличие A/B в drop — один эпизод из128, что не даёт оснований
объявлять победителя по этому короткому запуску с одним seed. В A насыщение
средних действий практически сохранилось: near-limit0.86934 →0.86882.
Эти smoke-проверки подтверждают корректность старта и продолжения, но не
сходимость и не исправление падений. Аудит скоростей остаётся отдельным
ограничением интерпретации; физика в этой кампании не менялась.

16 CPU-тестов verification/continuation/std-only и py_compile прошли.
Прогоны завершены, дополнительных обучений не запущено.
