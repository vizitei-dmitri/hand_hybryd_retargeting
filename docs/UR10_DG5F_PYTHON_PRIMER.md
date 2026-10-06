# Синтаксис Python и библиотеки в коде среды UR10e + DG-5F

Это разбор к [гайду](UR10_DG5F_ENV_GUIDE.md): не «что делает среда», а «что написано в
коде и как это читать». Только те конструкции и библиотеки, которые реально есть в
пакете `lerobot_robot_ur10_dg5f`.

- **Часть 1** — `env.py` (шаг 8) построчно. Начните с неё.
- **Часть 2** — справочник по синтаксису Python.
- **Части 3–8** — библиотеки: numpy, scipy, gymnasium, MuJoCo, LeRobot, pytest.
- **Часть 9** — как разбираться с незнакомым кодом самому.

## Как запускать примеры

Блоки, которые начинаются с `# пример`, можно запустить. Делайте это в контейнере, в
оболочке из шага 0 гайда (после `source …` и `export …`). Проще всего обернуть код в
`python3 - <<'EOF'` … `EOF`:

```bash
python3 - <<'EOF'
import numpy as np
print(np.array([1.0, 2.0]) * 3)
EOF
```

Блоки, которые начинаются с `# фрагмент`, — куски нашего кода для чтения, их не
запускают отдельно. Комментарий `# → ...` справа показывает, что печатает строка.

---

## Часть 1. `env.py` построчно

### 1.1. Импорты

```python
# фрагмент env.py
from typing import Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

from .pose_math import rotvec_to_6d
from .ur10_dg5f import ACTION_KEYS, ARM_KEYS, HAND_KEYS, TCP_KEYS, Ur10Dg5f, vector_to_action
```

- `import gymnasium as gym` — подключить библиотеку и дать ей короткое имя `gym`.
  Дальше `gym.Env` значит «класс `Env` из gymnasium».
- `from gymnasium import spaces` — взять из библиотеки один модуль `spaces`.
- `from lerobot.utils.constants import OBS_IMAGES, OBS_STATE` — взять две строковые
  константы: `OBS_STATE == "observation.state"`, `OBS_IMAGES == "observation.images"`.
  Константы вместо строк в коде нужны, чтобы не ошибиться в написании.
- `from .pose_math import ...` — **точка** значит «из этого же пакета»: файл
  `pose_math.py`, который лежит рядом. Это относительный импорт.
- `from typing import Callable` — тип «функция», нужен только для подсказки типа (1.4).

### 1.2. Константа модуля

```python
# фрагмент env.py
STATE_DIM = 3 + 6 + len(ARM_KEYS) + len(HAND_KEYS)
```

Переменная на уровне файла, имя заглавными — по договорённости это константа.
`len(...)` — длина кортежа: `ARM_KEYS` — 6 имён, `HAND_KEYS` — 20. Итого 35.

### 1.3. Объявление класса

```python
# фрагмент env.py
class Ur10Dg5fEnv(gym.Env):
    """Absolute 26-value actions: TCP pose (6) and DG5F joints in degrees (20)."""

    metadata = {"render_modes": []}
```

- `class Ur10Dg5fEnv(gym.Env):` — новый класс **на основе** `gym.Env`
  (наследование, 2.4). Мы получаем всё, что gymnasium уже умеет, и дописываем свои
  `reset` и `step`. Именно поэтому любой RL-код примет наш объект как «среду».
- Строка в тройных кавычках сразу под `class` — docstring, описание. Его показывает
  `help(Ur10Dg5fEnv)`.
- `metadata = {...}` — атрибут класса (общий для всех объектов), его требует
  gymnasium. Пустой список режимов рендера значит «картинку окна среда не рисует».

### 1.4. Конструктор `__init__`

```python
# фрагмент env.py
    def __init__(
        self,
        robot: Ur10Dg5f,
        fps: float = 10.0,
        episode_s: float = 30.0,
        wait_for_scene_reset: Callable[[], None] | None = None,
    ):
        super().__init__()
        self.robot = robot
        self.period_s = 1.0 / fps
        self.max_steps = round(episode_s * fps)
        self.wait_for_scene_reset = wait_for_scene_reset
        if not robot.is_connected:
            robot.connect()
```

- `__init__` вызывается, когда пишут `Ur10Dg5fEnv(robot)`: он «собирает» новый объект.
- `self` — сам этот объект. `self.robot = robot` — запомнить робота внутри объекта,
  чтобы потом в `step` написать `self.robot.send_action(...)`.
- `robot: Ur10Dg5f` — подсказка типа: «сюда ждут робота». Python её не проверяет, она
  для человека и редактора (2.3).
- `fps: float = 10.0` — параметр со значением по умолчанию: можно не передавать.
- `Callable[[], None] | None = None` — «функция без аргументов, которая ничего не
  возвращает, **или** `None`». На реальном роботе сюда передают функцию «ждать, пока
  оператор вернёт объект на место»; в симуляции не передают ничего.
- `super().__init__()` — вызвать конструктор родителя `gym.Env`, чтобы он тоже
  подготовил свои поля.
- `round(episode_s * fps)` — 30 с × 10 тактов/с = 300 тактов в попытке.
- `robot.is_connected` — без скобок: это свойство (`@property`, 2.5), а не метод.

```python
# фрагмент env.py
        self.action_space = spaces.Box(-np.inf, np.inf, shape=(len(ACTION_KEYS),), dtype=np.float32)
        observation_spaces = {
            OBS_STATE: spaces.Box(-np.inf, np.inf, shape=(STATE_DIM,), dtype=np.float32),
        }
        for name, shape in robot.observation_features.items():
            if isinstance(shape, tuple):
                observation_spaces[f"{OBS_IMAGES}.{name}"] = spaces.Box(0, 255, shape=shape, dtype=np.uint8)
        self.observation_space = spaces.Dict(observation_spaces)
        self._step_count = 0
```

- `spaces.Box(низ, верх, shape=..., dtype=...)` — описание «какие массивы бывают»:
  здесь действие — 26 чисел float32 без границ (`-np.inf`…`np.inf`). Это не данные, а
  **паспорт** данных: RL-библиотека читает его, чтобы знать размер выхода нейросети (5.2).
- `shape=(26,)` — кортеж из одного числа; запятая обязательна: `(26)` — просто число 26.
- `{OBS_STATE: ...}` — словарь «ключ: значение» (2.6).
- `for name, shape in robot.observation_features.items():` — пройти по парам
  «ключ, значение» словаря. У робота значения — либо `float` (обычное число), либо
  кортеж `(240, 320, 3)` для камеры.
- `isinstance(shape, tuple)` — «это кортеж?». Так отделяются камеры от чисел.
- `f"{OBS_IMAGES}.{name}"` — f-строка (2.7): подставит значения в фигурные скобки, получится
  `"observation.images.cam_front"`.
- `spaces.Dict({...})` — паспорт наблюдения из нескольких частей.
- `self._step_count` — подчёркивание в начале имени значит «внутреннее, снаружи не
  трогать» (это договорённость, не запрет).

### 1.5. `reset`

```python
# фрагмент env.py
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.robot.go_home()
        if self.wait_for_scene_reset is not None:
            self.wait_for_scene_reset()
        self._step_count = 0
        return self._observation(), {}
```

- `*` в списке параметров: всё после него передаётся **только по имени**:
  `env.reset(seed=1)` работает, `env.reset(1)` — ошибка. Так требует gymnasium.
- `super().reset(seed=seed)` — родитель настраивает генератор случайных чисел.
- `self.wait_for_scene_reset()` — со скобками: вызвать функцию, которую передали в
  конструктор.
- `return self._observation(), {}` — вернуть **кортеж из двух** значений: наблюдение и
  пустой словарь `info`. Снаружи их «распаковывают»: `obs, info = env.reset()` (2.6).

### 1.6. `step`

```python
# фрагмент env.py
    def step(self, action):
        sent = self.robot.send_action(vector_to_action(np.asarray(action, dtype=np.float64)))
        self.robot.wait_next_period(self.period_s)
        observation = self._observation()
        self._step_count += 1
        protective_stop = self.robot.protective_stop()
        info = {
            "sent_action": np.asarray([sent[key] for key in ACTION_KEYS], dtype=np.float64),
            "protective_stop": protective_stop,
        }
        return observation, 0.0, protective_stop, self._step_count >= self.max_steps, info
```

Первую строку читайте изнутри наружу:

1. `np.asarray(action, dtype=np.float64)` — что бы ни пришло (список, массив float32),
   сделать массив numpy из float64 (3.1).
2. `vector_to_action(...)` — функция из шага 7: массив 26 чисел → словарь
   `{"tcp.x": ..., ..., "rj_dg_5_4.pos": ...}`, потому что `Robot` LeRobot ждёт словарь.
3. `self.robot.send_action(...)` — отправить; вернёт, что реально ушло после ограничителей.

Дальше:

- `self._step_count += 1` — то же, что `self._step_count = self._step_count + 1`.
- `[sent[key] for key in ACTION_KEYS]` — генератор списка (2.6): «для каждого ключа
  взять `sent[key]`», обратно из словаря в список из 26 чисел.
- `return a, b, c, d, e` — пять значений: наблюдение, награда `0.0`, `terminated`,
  `truncated`, `info`. Это контракт gymnasium (5.1).
- `self._step_count >= self.max_steps` — сравнение даёт `True`/`False`: «время вышло?».

### 1.7. `_observation`

```python
# фрагмент env.py
    def _observation(self):
        raw = self.robot.get_observation()
        tcp = np.asarray([raw[key] for key in TCP_KEYS])
        state = np.concatenate([
            tcp[:3],
            rotvec_to_6d(tcp[3:]),
            [raw[key] for key in ARM_KEYS],
            [raw[key] for key in HAND_KEYS],
        ]).astype(np.float32)
        observation = {OBS_STATE: state}
        for key, value in raw.items():
            if isinstance(value, np.ndarray) and value.ndim == 3:
                observation[f"{OBS_IMAGES}.{key}"] = value
        return observation
```

- `tcp[:3]` — срез: элементы 0, 1, 2 (позиция). `tcp[3:]` — с 3-го до конца (поворот).
  Подробно — 2.8.
- `np.concatenate([...])` — склеить несколько кусков в один массив: 3 + 6 + 6 + 20 = 35.
- `.astype(np.float32)` — перевести в 32-битные числа: так хранят данные для нейросетей.
- `value.ndim == 3` — у массива 3 измерения (высота, ширина, цвет) — значит, это кадр
  камеры.

### 1.8. Попробуйте вживую (после шага 8)

```python
# пример: среда шага 8 вживую
import pathlib
from lerobot_robot_ur10_dg5f import Ur10Dg5f, Ur10Dg5fConfig
from lerobot_robot_ur10_dg5f.env import Ur10Dg5fEnv

config = Ur10Dg5fConfig(id="try", calibration_dir=pathlib.Path("/tmp/try"),
                        mjcf_path="/workspace/models/ur10e_dg5f/scene.xml", sim_cameras=("cam_front",))
robot = Ur10Dg5f(config)
env = Ur10Dg5fEnv(robot, fps=10, episode_s=2)

print(env.action_space)                       # → Box(-inf, inf, (26,), float32)
observation, info = env.reset()
print(list(observation))                      # → ['observation.state', 'observation.images.cam_front']
print(observation["observation.state"].shape) # → (35,)
z_start = observation["observation.state"][2] # state[0:3] — позиция TCP, [2] — высота z

target = robot.last_command.copy()            # 26 чисел: «стой, где стоишь»
target[2] += 0.05                             # ...но на 5 см выше
for tick in range(20):                        # 20 тактов = 2 с
    observation, reward, terminated, truncated, info = env.step(target)
print(round(float(observation["observation.state"][2] - z_start), 3))  # → 0.05  поднялись на 5 см
print(reward, terminated, truncated)          # → 0.0 False True
env.close()
```

Последняя строка: `truncated` стал `True`, потому что `episode_s=2` — это ровно 20 тактов.

---

## Часть 2. Синтаксис Python

### 2.1. Импорты

| Запись | Что значит |
| --- | --- |
| `import numpy as np` | весь модуль под коротким именем |
| `from numpy import linalg` | одно имя из модуля |
| `from .safety import clip_hand` | из соседнего файла этого же пакета |
| `import mujoco` внутри функции | «ленивый» импорт: выполнится только при вызове (так в `backend.py`, чтобы без MuJoCo импорт пакета не падал) |

### 2.2. Функции

```python
# пример: функции, значения по умолчанию, аргументы по имени
def clip(value, low=0.0, high=1.0):
    """Ограничить value отрезком [low, high]."""
    return min(max(value, low), high)

print(clip(5))                 # → 1.0
print(clip(5, high=10))        # → 5      (high передан по имени)

def reset(*, seed=None):       # всё после * — только по имени
    return seed

print(reset(seed=3))           # → 3

square = lambda x: x * x       # lambda — короткая безымянная функция
print(square(4))               # → 16
```

`lambda` в нашем коде: `guard=lambda proposed: ...` — «вот функция, которую позови сам»
(шаг 11), и `lambda: {BROKEN_PINKY_JOINT: 0.0}` в конфиге (2.5).

### 2.3. Подсказки типов

`x: float`, `-> np.ndarray`, `dict[str, float]`, `tuple[int, int]`, `str | None`,
`Callable[[dict], np.ndarray]`. Python их **не проверяет** при запуске. Они нужны
человеку и редактору (VS Code подсвечивает ошибки и подсказывает методы).

```python
# пример: подсказки типов не мешают передать что угодно
def half(x: float) -> float:
    return x / 2

print(half(3))       # → 1.5   (передали int — Python не возражает)
```

### 2.4. Классы, `self`, наследование, `super()`

```python
# пример: класс, объект, наследование
class Counter:
    start = 0                      # атрибут класса: общий для всех объектов

    def __init__(self, step):      # конструктор: Counter(2) вызывает его
        self.step = step           # атрибут объекта: у каждого свой
        self.value = self.start

    def tick(self):                # метод: self — это сам объект
        self.value += self.step
        return self.value

class LoudCounter(Counter):        # наследник: всё от Counter + своё
    def tick(self):
        value = super().tick()     # вызвать версию родителя
        print("tick ->", value)
        return value

c = LoudCounter(step=2)
c.tick()                           # → tick -> 2
c.tick()                           # → tick -> 4
print(isinstance(c, Counter))      # → True
```

В нашем коде: `Ur10Dg5f(Robot)`, `Ur10Dg5fEnv(gym.Env)`, `ResidualInterventionEnv(gym.Wrapper)`,
`Ur10Dg5fConfig(RobotConfig)`. Везде одна идея: библиотека задаёт «бланк» с
обязательными методами, а мы его заполняем.

### 2.5. Декораторы: `@property`, `@cached_property`, `@dataclass`, `@classmethod`

Декоратор — строка `@что-то` над функцией или классом. Она берёт функцию (класс) и
возвращает изменённую версию.

```python
# пример: @property и @cached_property
from functools import cached_property

class Robot:
    def __init__(self):
        self.connected = False

    @property
    def is_connected(self):        # читается как поле, без скобок
        return self.connected

    @cached_property
    def features(self):            # считается один раз, потом берётся готовое
        print("считаю признаки")
        return {"tcp.x": float}

r = Robot()
print(r.is_connected)              # → False
r.features                         # → считаю признаки
r.features                         # (второй раз ничего не печатает)
```

```python
# пример: @dataclass — класс-«запись» без ручного __init__
from dataclasses import dataclass, field

@dataclass
class Config:
    backend: str = "mujoco"
    max_step_m: float = 0.010
    disabled: dict = field(default_factory=lambda: {"rj_dg_5_1": 0.0})

    def __post_init__(self):       # вызывается сразу после автоматического __init__
        if self.max_step_m <= 0:
            raise ValueError("max_step_m must be positive")

print(Config())                    # → Config(backend='mujoco', max_step_m=0.01, disabled={'rj_dg_5_1': 0.0})
print(Config(max_step_m=0.005).max_step_m)   # → 0.005
```

`field(default_factory=...)` нужен для изменяемых значений по умолчанию (словарь,
список): иначе все объекты делили бы **один** словарь. `@dataclass(frozen=True)`
(`ResidualScale`) запрещает менять поля после создания.

`@RobotConfig.register_subclass("ur10_dg5f")` — тоже декоратор: он кладёт наш класс в
реестр LeRobot под именем `ur10_dg5f` (7.1).

```python
# пример: @classmethod — «другой конструктор»
class Synergies:
    def __init__(self, components):
        self.components = components

    @classmethod
    def fit(cls, data):            # cls — сам класс, не объект
        return cls(components=sorted(data)[:2])

s = Synergies.fit([5, 1, 3])       # вызываем у класса, получаем объект
print(s.components)                # → [1, 3]
```

Так устроены `HandSynergies.fit(...)` и `HandSynergies.load(...)` в шаге 9.

### 2.6. Списки, кортежи, словари, генераторы, распаковка

```python
# пример: коллекции
joints = ["rj_dg_1_1", "rj_dg_1_2"]          # список: можно менять
shape = (240, 320, 3)                        # кортеж: нельзя менять
pose = {"tcp.x": 0.1, "tcp.y": 0.5}          # словарь: ключ -> значение

print(pose["tcp.x"])                         # → 0.1
for key, value in pose.items():              # пары ключ-значение
    print(key, value)                        # → tcp.x 0.1 / tcp.y 0.5

squares = [x * x for x in range(4)]          # генератор списка
print(squares)                               # → [0, 1, 4, 9]
keys = {name: 0.0 for name in joints}        # генератор словаря
print(keys)                                  # → {'rj_dg_1_1': 0.0, 'rj_dg_1_2': 0.0}

a, b = (1, 2)                                # распаковка кортежа
obs, reward, terminated, truncated, info = ("o", 0.0, False, False, {})
print(dict(zip(["x", "y"], [1, 2])))         # zip: пары из двух списков → {'x': 1, 'y': 2}
```

В нашем коде: `{key: float(value) for key, value in zip(ACTION_KEYS, vector, strict=True)}`
— словарь из двух списков; `strict=True` выдаст ошибку, если длины не совпали.

### 2.7. f-строки

```python
# пример: f-строки
name, value = "cam_front", 3.14159
print(f"observation.images.{name}")          # → observation.images.cam_front
print(f"{value:.2f}")                        # → 3.14   (2 знака после точки)
print(f"{'rj_dg_1_1'}_ctrl")                 # → rj_dg_1_1_ctrl
```

### 2.8. Срезы

```python
# пример: срезы
a = [10, 11, 12, 13, 14, 15]
print(a[:3])      # → [10, 11, 12]   первые три (позиция TCP)
print(a[3:])      # → [13, 14, 15]   с 3-го до конца (поворот)
print(a[3:5])     # → [13, 14]       с 3-го по 4-й
print(a[-1])      # → 15             последний
```

Действие из 26 чисел режут так: `action[:6]` — поза TCP, `action[6:]` — 20 суставов
кисти; внутри позы `[:3]` — позиция, `[3:6]` — поворот.

### 2.9. Ошибки: `raise`, `try/except`, `assert`

```python
# пример: ошибки
def check(step):
    if step <= 0:
        raise ValueError("step must be positive")   # остановить с понятным сообщением
    return step

try:
    check(-1)
except ValueError as error:                         # поймать и обработать
    print("поймали:", error)                        # → поймали: step must be positive

assert check(2) == 2                                # «должно быть так», иначе AssertionError
```

### 2.10. `with`, `Protocol`, `Enum`

- `with self._lock:` — «взять замок на время блока и обязательно отпустить» (шаг 11).
  `with np.load(path) as data:` — открыть файл и гарантированно закрыть.
- `class Backend(Protocol):` — описание «какие методы должны быть». Наследоваться от
  него не обязательно: подходит любой объект с такими методами («утиная типизация»).
  Так `MujocoBackend` и `RealBackend` взаимозаменяемы.
- `TeleopEvents.IS_INTERVENTION` — элемент перечисления (`Enum`): именованная константа,
  её используют как ключ словаря событий.

```python
# пример: Enum как ключ словаря
from enum import Enum

class Events(Enum):
    IS_INTERVENTION = "is_intervention"
    SUCCESS = "success"

events = {Events.IS_INTERVENTION: True}
print(events.get(Events.SUCCESS, False))     # → False  (.get с запасным значением)
```

### 2.11. Потоки (шаг 11)

```python
# пример: поток и замок
import threading, time

target = [0]
lock = threading.Lock()
running = True

def loop():                                  # крутится отдельно, как поток servoL
    while running:
        with lock:
            current = target[0]
        time.sleep(0.01)

thread = threading.Thread(target=loop, daemon=True)
thread.start()
with lock:
    target[0] = 42                           # основной поток меняет цель под замком
running = False
thread.join()                                # дождаться завершения
print("поток остановлен, цель", target[0])   # → поток остановлен, цель 42
```

---

## Часть 3. numpy

numpy — массивы чисел и математика над ними сразу, без циклов. Документация:
<https://numpy.org/doc/stable/user/absolute_beginners.html>.

```python
# пример: массивы numpy
import numpy as np

a = np.array([1.0, 2.0, 3.0])
print(a.shape, a.dtype)               # → (3,) float64
print(a * 2 + 1)                      # → [3. 5. 7.]     действие над каждым элементом
print(np.linalg.norm([3.0, 4.0]))     # → 5.0            длина вектора
print(np.clip([-5, 0.5, 9], 0, 1))    # → [0.  0.5 1. ]  обрезать в [0, 1]
print(np.concatenate([[1, 2], [3]]))  # → [1 2 3]

m = np.eye(3)                         # единичная матрица 3×3
print(m @ a)                          # → [1. 2. 3.]     @ — умножение матриц
print(np.zeros((2, 3)).shape)         # → (2, 3)
print(np.deg2rad(180.0))              # → 3.141592653589793
print(np.all(np.isfinite([1.0, np.nan])))  # → False  есть ли NaN
```

**Индексы-массивы** (так в MuJoCo-бэкенде берут суставы руки из общего массива):

```python
# пример: выбрать элементы по списку индексов
import numpy as np
qpos = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
arm = np.array([0, 2, 4])
print(qpos[arm])                      # → [0.1 0.3 0.5]
qpos[arm] = 0.0                       # и записать туда же
print(qpos)                           # → [0.  0.2 0.  0.4 0. ]
```

**Срез — это окно, а не копия.** Поэтому в коде так много `.copy()`:

```python
# пример: зачем .copy()
import numpy as np
a = np.array([1.0, 2.0, 3.0, 4.0])
window = a[:2]
window[0] = 100.0
print(a)                # → [100.   2.   3.   4.]  исходный массив тоже изменился!
safe = a[:2].copy()
safe[0] = -1.0
print(a)                # → [100.   2.   3.   4.]  копия не трогает оригинал
```

`self._last_tcp = state.tcp.copy()` в шаге 7 — ровно поэтому: без копии «последняя
отправленная цель» менялась бы вместе с массивом, из которого её взяли.

`dtype`: `float64` — обычные числа для расчётов; `float32` — для нейросетей (вдвое
меньше памяти); `uint8` — пиксели 0–255.

---

## Часть 4. scipy: повороты

`scipy.spatial.transform.Rotation` — объект «поворот», который умеет переводить себя
между представлениями. Документация:
<https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html>.

```python
# пример: повороты на вашей home-позе
import numpy as np
from scipy.spatial.transform import Rotation as R

home = R.from_rotvec([3.1308, -0.0012, -0.0014])   # из шага 7: поворот TCP в home
print(np.degrees(home.magnitude()).round(1))       # → 179.4      угол поворота
print(home.as_matrix()[:, 2].round(3))             # → [-0.001 -0.011 -1.   ]  ось z инструмента смотрит вниз

quarter = R.from_euler("z", 90, degrees=True)      # 90° вокруг z
print(quarter.apply([1, 0, 0]).round(3))           # → [0. 1. 0.]  повернуть вектор
both = quarter * quarter                           # композиция: сначала правый, потом левый
print(np.degrees(both.magnitude()).round(1))       # → 180.0
print(np.degrees((both * quarter.inv()).magnitude()).round(1))  # → 90.0   .inv() — обратный
```

В `pose_math.py`: `R.from_rotvec(delta[3:6]) * R.from_rotvec(base[3:6])` — «сначала
поворот базы, потом поворот смещения», а `.as_rotvec()` переводит результат обратно в
3 числа для позы UR.

---

## Часть 5. gymnasium

Документация: <https://gymnasium.farama.org/introduction/basic_usage/> и
<https://gymnasium.farama.org/introduction/create_custom_env/>.

### 5.1. Контракт среды

Любая среда gymnasium — это класс с двумя главными методами:

- `reset()` → `(observation, info)` — начать попытку;
- `step(action)` → `(observation, reward, terminated, truncated, info)` — один такт.

Вот самая маленькая среда, на которой видно весь контракт:

```python
# пример: игрушечная среда — дойти счётчиком до 3
import gymnasium as gym
import numpy as np
from gymnasium import spaces

class CountToThree(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-10.0, 10.0, shape=(1,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.value = 0.0
        self.ticks = 0
        return np.array([self.value], dtype=np.float32), {}

    def step(self, action):
        self.value += float(action[0])
        self.ticks += 1
        success = self.value >= 3.0
        reward = 1.0 if success else 0.0
        observation = np.array([self.value], dtype=np.float32)
        return observation, reward, success, self.ticks >= 10, {}

env = CountToThree()
observation, info = env.reset()
for tick in range(10):
    observation, reward, terminated, truncated, info = env.step(np.array([1.0]))
    print(tick, observation, reward, terminated, truncated)
    if terminated or truncated:
        break
# → 0 [1.] 0.0 False False
# → 1 [2.] 0.0 False False
# → 2 [3.] 1.0 True False
```

`Ur10Dg5fEnv` устроена точно так же, только вместо счётчика — робот.

### 5.2. `spaces`: паспорт данных

```python
# пример: Box и Dict
import numpy as np
from gymnasium import spaces

box = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
print(box.sample().shape)                         # → (3,)   случайный пример
print(box.contains(np.zeros(3, dtype=np.float32)))  # → True  подходит ли массив
image = spaces.Box(0, 255, shape=(240, 320, 3), dtype=np.uint8)
both = spaces.Dict({"observation.state": box, "observation.images.cam": image})
print(sorted(both.sample().keys()))               # → ['observation.images.cam', 'observation.state']
```

### 5.3. `gym.Wrapper`: обёртка (шаг 10)

```python
# пример: обёртка меняет вид действий, но зовёт ту же step()
import gymnasium as gym
import numpy as np
from gymnasium import spaces

class Inner(gym.Env):
    action_space = spaces.Box(-10, 10, shape=(1,))
    observation_space = spaces.Box(-10, 10, shape=(1,))
    def reset(self, *, seed=None, options=None):
        return np.zeros(1), {}
    def step(self, action):
        return np.array(action), 0.0, False, False, {"got": float(action[0])}

class Halve(gym.Wrapper):                   # снаружи: действие в [-1, 1]
    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Box(-1, 1, shape=(1,))
    def step(self, action):
        return self.env.step(np.asarray(action) * 0.5)   # внутрь уходит половина

env = Halve(Inner())
env.reset()
print(env.step([1.0])[4])                   # → {'got': 0.5}
```

`ResidualInterventionEnv` — такая же обёртка: снаружи 9 чисел поправки, внутрь уходят
26 абсолютных чисел. `self.env` — обёрнутая среда, `self.env.unwrapped` — самая
внутренняя (без всех обёрток).

### 5.4. `check_env`

`gymnasium.utils.env_checker.check_env(env)` прогоняет среду на соответствие контракту:
форма наблюдений совпадает с паспортом, `reset(seed=...)` воспроизводим и так далее.
Предупреждения про `-inf`/`inf` у нас ожидаемы (шаг 8).

---

## Часть 6. MuJoCo

Документация Python-биндингов: <https://mujoco.readthedocs.io/en/stable/python.html>.

- `MjModel` — **неизменяемое** описание сцены: тела, суставы, актуаторы, массы.
- `MjData` — **состояние**: текущие углы (`qpos`), скорости (`qvel`), команды
  актуаторам (`ctrl`), вычисленные позиции тел и сайтов.
- `mj_forward` — пересчитать позиции по текущим углам, время не идёт.
- `mj_step` — шаг физики, время идёт.

```python
# пример: модель, данные, доступ по имени
import mujoco

model = mujoco.MjModel.from_xml_path("/workspace/models/ur10e_dg5f/scene.xml")
data = mujoco.MjData(model)
print(model.nu, model.opt.timestep)                  # → 26 0.002   актуаторов, шаг физики

joint = model.joint("elbow_joint")                   # доступ по имени
print(joint.qposadr[0])                              # → 2          где угол лежит в data.qpos
data.qpos[joint.qposadr[0]] = 1.0                    # согнуть локоть на 1 рад
mujoco.mj_forward(model, data)                       # пересчитать позиции
site = model.site("attachment_site").id
print(data.site_xpos[site].round(3))                 # позиция фланца в мире (3 числа)
print(data.site_xmat[site].reshape(3, 3).shape)      # → (3, 3)    его поворот, матрица

actuator = model.actuator("elbow").id
data.ctrl[actuator] = 1.2                            # команда позиционному актуатору
mujoco.mj_step(model, data, nstep=50)                # 50 шагов × 2 мс = 0.1 с
print(data.time)                                     # → 0.1 (примерно)
```

`MjSpec` (шаг 6) — редактируемая версия описания до компиляции: поменяли атрибуты →
`spec.compile()` → получили `MjModel`.

---

## Часть 7. LeRobot

### 7.1. Реестр конфигов

```python
# пример: как LeRobot узнаёт наш тип робота (после шага 5)
from lerobot.robots import RobotConfig
import lerobot_robot_ur10_dg5f                        # импорт выполняет декоратор регистрации

print("ur10_dg5f" in RobotConfig.get_known_choices())  # → True
config = lerobot_robot_ur10_dg5f.Ur10Dg5fConfig(max_step_m=0.005)
print(config.type, config.max_step_m)                  # → ur10_dg5f 0.005
```

### 7.2. Контракт `Robot`

`lerobot/robots/robot.py` задаёт обязательные методы: `connect`, `get_observation`,
`send_action`, `disconnect`, свойства `observation_features`, `action_features` и др.
`lerobot-record` и прочие команды LeRobot зовут только их, поэтому им всё равно, что
за робот внутри.

```python
# пример: признаки робота (после шага 7)
import pathlib
from lerobot_robot_ur10_dg5f import Ur10Dg5f, Ur10Dg5fConfig

robot = Ur10Dg5f(Ur10Dg5fConfig(id="try", calibration_dir=pathlib.Path("/tmp/try"), sim_cameras=()))
print(list(robot.action_features)[:3])        # → ['tcp.x', 'tcp.y', 'tcp.z']
print(len(robot.action_features))             # → 26
print(robot.is_connected)                     # → False  (признаки доступны и без подключения)
```

### 7.3. Константы и события

- `OBS_STATE`, `OBS_IMAGES` из `lerobot.utils.constants` — стандартные ключи
  наблюдений в датасетах LeRobot.
- `TeleopEvents` из `lerobot.teleoperators.utils` — перечисление событий телеоператора:
  `IS_INTERVENTION`, `SUCCESS`, `TERMINATE_EPISODE`, `RERECORD_EPISODE`.

---

## Часть 8. pytest

Документация: <https://docs.pytest.org/en/stable/getting-started.html>.

- Тест — функция с именем `test_...` в файле `test_...py`. Внутри — `assert`.
- **Фикстура** — функция с `@pytest.fixture`, которая готовит что-то для теста.
  Тест просит её **по имени параметра**: `def test_x(make_sim_robot):`. Общие фикстуры
  лежат в `conftest.py`, pytest находит их сам. Встроенные: `tmp_path` (временный
  каталог), `monkeypatch` (временно подменить модуль или атрибут).
- `@pytest.mark.parametrize("offset", [...])` — один тест, запущенный для каждого
  значения из списка.
- `pytest.raises(ValueError)` — тест проходит, только если внутри была эта ошибка.
- `pytest.importorskip("mujoco")` — пропустить тест, если библиотеки нет.

```python
# пример: всё сразу в одном файле теста (pytest запускается изнутри примера)
import pathlib, subprocess, sys, tempfile

test_code = '''
import pytest

@pytest.fixture
def numbers():
    return [3, 1, 2]

def test_sorted(numbers):
    assert sorted(numbers) == [1, 2, 3]

@pytest.mark.parametrize("value", [1, 2, 3])
def test_positive(value):
    assert value > 0

def test_error():
    with pytest.raises(ZeroDivisionError):
        1 / 0
'''
folder = pathlib.Path(tempfile.mkdtemp())
(folder / "test_demo.py").write_text(test_code)
result = subprocess.run([sys.executable, "-m", "pytest", "-q", str(folder)], capture_output=True, text=True)
print(result.stdout.strip().splitlines()[-1])     # → 5 passed in ...s
```

Полезные флаги: `-q` коротко, `-x` остановиться на первой ошибке, `-k имя` запустить
только тесты с этим словом в имени, `-s` показывать `print` из тестов.

---

## Часть 9. Как разбираться с незнакомым кодом самому

- **Посмотреть, что за объект:** `type(x)`, `x.shape`, `x.dtype`, `list(d)` для словаря.
- **Почитать справку:** `help(spaces.Box)`, `help(np.concatenate)` — прямо в `python3`.
- **Посмотреть, что умеет объект:** `dir(env)`.
- **Найти определение:** в VS Code — Ctrl+клик по имени (F12).
- **Поставить точку остановки:** вставьте `breakpoint()` в код и запустите тест с
  `-s`: выполнение остановится, можно печатать переменные (`p имя`), идти дальше (`n`),
  продолжить (`c`).
- **Интерактивная оболочка:** `python3` в контейнере, импортируйте модуль и пробуйте
  по строчке. IPython в образе нет, хватает обычного `python3`.

```python
# пример: исследовать незнакомый объект
import numpy as np
from gymnasium import spaces

box = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
print(type(box).__name__)                  # → Box
print(box.low, box.high)                   # → [-1. -1.] [1. 1.]
print([n for n in dir(box) if not n.startswith("_")][:6])   # первые публичные методы и поля
```
