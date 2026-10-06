# Среда UR10e + DG-5F для диплома — пошаговый гайд

Гайд ведёт от пустой ветки `feature/ur10-dg5f-env` до среды, которую требует план
(вариант B): робот UR10e + DG-5F в контракте LeRobot, gym-среда поверх него и обёртка
«residual + перехват оператора» с наградой RLIF.

## Как пользоваться

Каждый шаг устроен одинаково и читается сверху вниз:

- **Зачем** — какую часть плана закрывает шаг.
- **Задание** — что сделать.
- **Решение: команды** — сразу под заданием. Готовые блоки для терминала: каждый файл
  создаётся целиком командой `cat > путь <<'EOF' … EOF`, в конце — команды проверки.
  Вставляйте блоки по порядку, сверху вниз.
- **Проверка** — что должны напечатать команды.
- **Откуда это и где найти самому** — в конце шага: источники (документация, исходники
  LeRobot и MuJoCo, файлы этого репо, статьи) и то, как к решению можно прийти
  самостоятельно. Это учебная часть: читайте её, чтобы понимать, что именно вставили.

**Непонятен синтаксис или библиотека?** Разбор кода — в [UR10_DG5F_PYTHON_PRIMER.md](UR10_DG5F_PYTHON_PRIMER.md): `env.py` построчно, справочник по Python, numpy, scipy, gymnasium, MuJoCo, LeRobot и pytest, у каждой темы — запускаемый пример.

Каждый блок команд начинается с комментария, где его выполнять: `# на хосте` (из корня
репо) или `# в контейнере` (оболочка из шага 0). Пути внутри блоков для контейнера
абсолютные (`/workspace/...`), поэтому блок работает из любого каталога. Кавычки в
`<<'EOF'` важны: с ними bash не подставляет `$...` внутри файла (например, `$base` в
`setup.cfg`).

Как это проверено (2026-10-06, Docker-образ `dg5f-lerobot-retargeting:humble`, MuJoCo
3.3.7, LeRobot 0.4.4, модель коллеги): все 42 блока `# в контейнере` взяты из этого
самого файла и выполнены по порядку одним скриптом в чистом контейнере на копии репо.
Каждый шаг напечатал ожидаемое, весь пакет — **54 passed**, а созданные файлы побайтно
совпали с эталоном. Блоки `# на хосте` (копирование модели, `stack.sh`) так не
прогонялись. Реальный UR10e и URSim я не запускал: реальный бэкенд проверен только
против подделок `ur_rtde` и мок-кисти (шаг 11).

### Общий приём «как найти самому»

Почти все ответы в этом гайде найдены тремя способами. Пользуйтесь ими:

1. **Читать исходники установленной библиотеки.** Документация отстаёт, код — нет:
   ```bash
   python3 -c "import lerobot, os; print(os.path.dirname(lerobot.__file__))"
   grep -rn "def register_third_party_plugins" $(python3 -c "import lerobot, os; print(os.path.dirname(lerobot.__file__))")
   ```
2. **Читать docstring прямо из Python**, особенно у C++-биндингов вроде `ur_rtde`:
   `python3 -c "import rtde_control; help(rtde_control.RTDEControlInterface.servoL)"`.
3. **Маленький эксперимент вместо догадки.** Скрипт на 20 строк, который печатает
   одно число (дрейф, силу актуатора, кватернион тела), решает спор быстрее чтения
   документации. Три ловушки модели на шаге 6 найдены именно так.

## Что строим

```text
 SAC: residual, 6+k чисел в [-1, 1]      оператор в Quest (позже: lerobot_teleoperator_quest)
              │                                        │
              ▼                                        ▼
 ResidualInterventionEnv   a_exec = a_base ⊕ β·clip(Δa) или a_op; флаги, награда, метки residual
              │  26 чисел: TCP (6) + кисть (20, градусы)
              ▼
 Ur10Dg5fEnv (gym.Env)     пространства, reset в home, темп fps, конец эпизода
              │
              ▼
 Ur10Dg5f (LeRobot Robot)  ограничители: рабочая зона, шаг TCP, пределы суставов кисти
              │
              ▼
 Backend:  MujocoBackend (отладка без железа)  |  RealBackend (ur_rtde + servo DG5F)
```

Почему не готовый `gym_manipulator` из LeRobot: его `RobotEnv`
(`lerobot/rl/gym_manipulator.py`) читает `robot.bus.motors` и делает reset через
`bus.sync_write`, то есть рассчитан на сервоприводы Feetech/Dynamixel. У UR10e шины
нет, поэтому среда своя, а контракт `Robot` и события `TeleopEvents` берём у LeRobot.
Тогда `lerobot-record` и `lerobot-train` потом заработают с теми же данными.

## Что уже есть и чего нет

| Есть | Где |
| --- | --- |
| LeRobot 0.4.4 | на хосте в `~/.local` и в Docker-образе |
| MuJoCo 3.3.7, scipy 1.15.3, gymnasium 1.3.0, pytest | только в Docker-образе (на хосте нет ни MuJoCo, ни pytest) |
| Плагин кисти `lerobot_robot_dg5f` | `src/lerobot_robot_dg5f/`: суставы, пределы, мок-бэкенд, servo-контроллер, ROS-мост |
| MJCF-модель UR10e + DG-5F | **не в этом репо**: `~/Documents/work/hand/tesollo_dg5f_mujoco/robot/` (репозиторий коллеги `VAlikV/tesollo_dg5f_mujoco`) |

| Нет | Комментарий |
| --- | --- |
| Кода UR10e | в репо ни одной строки; нужен `ur_rtde` (проверен 1.6.5) |
| Позы запястья с Quest | `ManoLandmarks` несёт только 21 точку; понадобится для телеоператора, не для этого гайда |

Модель коллеги — тоже UR10e, так что позы TCP симулятора и реального робота сравнимы.

---

## Шаг 0. Ветка и окружение

**Зачем.** Понять, где запускается код, до того как его писать.

**Задание.** Ветка уже создана (`git switch -c feature/ur10-dg5f-env` от `1f34f77`).
Поднимите контейнер и убедитесь, что в нём есть всё нужное.

### Решение: команды

На хосте:

```bash
# на хосте
bash scripts/stack.sh up        # поднять контейнер, если не поднят
bash scripts/stack.sh shell     # открыть оболочку внутри контейнера
```

Внутри контейнера, **в каждой новой оболочке**:

```bash
# в контейнере
source /opt/ros/humble/setup.bash
source /workspace/install/setup.bash
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MUJOCO_GL=egl
python3 -c "import mujoco, scipy, gymnasium, lerobot, pytest, lerobot_robot_dg5f; print('ok')"
```

Зачем каждая строка:

- `source ...` — `stack.sh shell` выполняет `docker compose exec lerobot_hand bash`
  (посмотрите: `grep -n 'shell)' -A3 scripts/stack.sh`). `exec` открывает новую
  оболочку **мимо entrypoint** образа, а ROS и `install/setup.bash` подключает именно
  entrypoint (`cat docker/ros_entrypoint.sh`). Без этих строк не импортируется
  `lerobot_robot_dg5f`, от которого зависит новый пакет.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` — так тесты запускает сам репо
  (`scripts/test_inside.sh`): ROS ставит свои pytest-плагины, их автозагрузка мешает.
- `MUJOCO_GL=egl` — рендер камер без окна. В `compose.yaml` стоит `glfw`, которому
  нужен дисплей; `osmesa` в образе не работает (проверено).

Чтобы не набирать это в каждой оболочке, можно один раз дописать в `~/.bashrc`
контейнера (сохранится, пока контейнер не пересоздан):

```bash
# в контейнере, один раз, по желанию
cat >> ~/.bashrc <<'EOF'
source /opt/ros/humble/setup.bash
[ -f /workspace/install/setup.bash ] && source /workspace/install/setup.bash
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MUJOCO_GL=egl
EOF
```

Строка `groups: cannot find name for group ID 110` при входе безвредна: `compose.yaml`
добавляет пользователю группу render по числовому GID (`group_add`), а в образе у этого
GID нет имени.

**Что видно из контейнера.** В `compose.yaml` смонтирован только сам репо
(`- .:/workspace`). Поэтому внутри контейнера `~` — это `/home/hand`, и ничего за
пределами репо (например, модели коллеги в `~/Documents/work/hand/...`) там нет.
Правило гайда: всё, что берёт файлы снаружи репо, и команды `stack.sh` — **на хосте**;
создание файлов пакета и тесты — **в контейнере**. Каждый блок команд начинается с
комментария `# на хосте` или `# в контейнере`. Файлы, созданные в контейнере, сразу
видны на хосте: это один и тот же каталог.

На ветку вместе с вами переехали незакоммиченные правки Isaac Lab (`isaaclab_ext/...`).
Они к этой ветке не относятся: коммитьте здесь только свои пути
(`git add src/lerobot_robot_ur10_dg5f models/ur10e_dg5f ...`), а не `git add -A`.

**Проверка.** Последняя команда печатает `ok`.

### Откуда это и где найти самому

Окружение описано в самом репо: `compose.yaml`
(сервис `lerobot_hand`, репо смонтирован в `/workspace`), `docker/Dockerfile`,
`docker/python-requirements.txt` (там закреплены `mujoco==3.3.7` и `lerobot==0.4.4`),
список команд — `bash scripts/stack.sh help`. Проверка «а где оно установлено» —
одна строка: на хосте `python3 -c "import mujoco"` даёт `ModuleNotFoundError`, в
контейнере — нет. Отсюда правило гайда: всё с MuJoCo и все тесты — в контейнере.

---

## Шаг 1. Импорт модели вручную

**Зачем.** Без железа отлаживать всё, что выше бэкенда, можно только в симуляторе.

**Задание.** Перенесите в репо MJCF-модель UR10e + DG-5F, загрузите её и выясните, в
какой системе координат она живёт.

### Решение: команды

Копия (рекомендую). **На хосте, из корня репо**, не в контейнере:

```bash
# на хосте
cd ~/Documents/work/hand_hybryd_retargeting
SRC=~/Documents/work/hand/tesollo_dg5f_mujoco/robot
mkdir -p models/ur10e_dg5f/assets
cp $SRC/scene.xml $SRC/ur10edg5f.xml models/ur10e_dg5f/
cp -r $SRC/assets/ur10e $SRC/assets/dg5f_right $SRC/assets/box models/ur10e_dg5f/assets/
git -C $SRC rev-parse --short HEAD     # запишите коммит, с которого скопировали (783b992)
```

`SRC=...` должен стоять в той же сессии терминала, что и `cp`: без него `$SRC`
пустой, и `cp` ищет `/scene.xml`.

Или submodule (на хосте, по HTTPS): модель остаётся чужим репо, правки в него — через
коллегу.

```bash
# на хосте, вместо копии
git submodule add https://github.com/VAlikV/tesollo_dg5f_mujoco.git third_party/tesollo_dg5f_mujoco
git -C third_party/tesollo_dg5f_mujoco checkout 783b992
# тогда в конфиге: mjcf_path="third_party/tesollo_dg5f_mujoco/robot/scene.xml"
# после клонирования репо на другой машине: git submodule update --init
```

Запись об источнике в `THIRD_PARTY.md`:

```bash
# в контейнере
cat >> /workspace/THIRD_PARTY.md <<'EOF'

## UR10e + DG5F MuJoCo model

`models/ur10e_dg5f/` is copied from https://github.com/VAlikV/tesollo_dg5f_mujoco
(commit 783b992, used with the author's permission). The DG5F meshes come from
tesollo/delto_m_ros2 (BSD-3-Clause); the UR10e model follows mujoco_menagerie.
EOF
```

**Проверка:**

```bash
# в контейнере
cd /workspace
python3 - <<'EOF'
import mujoco
m = mujoco.MjModel.from_xml_path("models/ur10e_dg5f/scene.xml")
d = mujoco.MjData(m); mujoco.mj_forward(m, d)
print("nu", m.nu, "cams", [m.camera(i).name for i in range(m.ncam)])
print("base quat", d.xquat[m.body("base").id])
EOF
```

Ожидается `nu 26`, камеры `cam_front` и `cam_side`, `base quat [0 0 0 -1]`.
Визуально, в отдельной оболочке без `MUJOCO_GL=egl`:
`python3 -m mujoco.viewer --mjcf=/workspace/models/ur10e_dg5f/scene.xml` (на хосте
перед этим `bash scripts/stack.sh gui-on`).

### Откуда это и где найти самому

- *Где модель.* `grep -rli ur10 ~/Documents/work --include=*.xml` находит
  `hand/tesollo_dg5f_mujoco/robot/ur10edg5f.xml`. Чей это код: `git -C ... log` и
  `git -C ... remote -v` → автор VAlikV, `github.com/VAlikV/tesollo_dg5f_mujoco`.
  Лицензии в репо нет, спросите у коллеги разрешения для диплома.
- *Какие файлы реально нужны.* `grep -o 'file="[^"]*"' scene.xml ur10edg5f.xml`
  показывает, что сцена ссылается только на `assets/ur10e`, `assets/dg5f_right` и
  `assets/box`. Папки `urdf/` (12 МБ), `assets/2f85` (19 МБ) и `assets/gripper` не
  используются. Копия без них весит 44 МБ вместо 76 МБ.
- *Система координат.* Маленький эксперимент: `print(data.xquat[model.body("base").id])`
  даёт `(0, 0, 0, -1)`, то есть тело `base` повёрнуто на π вокруг z. Почему так, написано
  в официальном описании UR (`Universal_Robots_ROS2_Description/urdf/ur_macro.xacro`,
  сустав `base_link-base_fixed_joint`, `rpy="0 0 ${pi}"`): *«as base_link is REP-103
  aligned … this is needed to correctly align 'base' with the 'Base' coordinate system
  of the UR controller»*. Значит, мир MuJoCo — это `base_link` из ROS, а
  `getActualTCPPose()` контроллера отдаёт позы в `base`: `(x, y, z)_UR = (−x, −y, z)_MuJoCo`.
  Это же объясняет `R_fix = diag(-1, -1, 1)` в `env_dg_ur.py` коллеги.
- *Submodule или копия.* Git Book, глава «Git Tools — Submodules»
  (<https://git-scm.com/book/en/v2/Git-Tools-Submodules>). Копию проще править: на шаге 6
  нужна камера на запястье, а тесты по умолчанию ищут модель в `models/ur10e_dg5f/`.
- *Где выполнять.* Модель лежит вне репо, а контейнер видит только репо (шаг 0). Поэтому
  копирование — **на хосте**, проверка загрузки — в контейнере.
- *SSH или HTTPS.* Адрес `git@github.com:...` требует SSH-ключа, зарегистрированного на
  GitHub. В контейнере ключа нет, отсюда `Permission denied (publickey)`. Публичный ли
  репо, проверяется без ключа: `git ls-remote https://github.com/VAlikV/tesollo_dg5f_mujoco.git`
  отвечает `783b992… HEAD`, значит, публичный, и для чтения хватает HTTPS.

---

## Шаг 2. Каркас пакета, и как Python и LeRobot его увидят

**Зачем.** Чтобы `--robot.type=ur10_dg5f` заработал в `lerobot-record` и прочих
командах LeRobot без правок самого LeRobot.

**Задание.** Создайте пакет `src/lerobot_robot_ur10_dg5f/` с `package.xml`, `setup.py`,
`setup.cfg`, пустым маркером `resource/lerobot_robot_ur10_dg5f` и `__init__.py`.

### Решение: команды

Структура пакета к концу гайда:

```text
src/lerobot_robot_ur10_dg5f/
├── package.xml
├── setup.py
├── setup.cfg
├── resource/lerobot_robot_ur10_dg5f      # пустой файл-маркер для ament
├── lerobot_robot_ur10_dg5f/
│   ├── __init__.py             растёт: шаг 2 → 5 → 7
│   ├── pose_math.py            шаг 3
│   ├── safety.py               шаг 4
│   ├── config_ur10_dg5f.py     шаг 5
│   ├── backend.py              шаг 6
│   ├── sim_backend.py          шаг 6
│   ├── ur10_dg5f.py            шаг 7
│   ├── env.py                  шаг 8
│   ├── synergies.py            шаг 9
│   ├── residual.py             шаг 10
│   └── real_backend.py         шаг 11
└── test/
    ├── conftest.py             шаг 2, общие фикстуры
    └── test_*.py               по одному на шаг
```

Каталоги и маркер:

```bash
# в контейнере
mkdir -p /workspace/src/lerobot_robot_ur10_dg5f/lerobot_robot_ur10_dg5f \
         /workspace/src/lerobot_robot_ur10_dg5f/test \
         /workspace/src/lerobot_robot_ur10_dg5f/resource
touch /workspace/src/lerobot_robot_ur10_dg5f/resource/lerobot_robot_ur10_dg5f
```

Файл `src/lerobot_robot_ur10_dg5f/package.xml`:

```bash
# в контейнере
cat > /workspace/src/lerobot_robot_ur10_dg5f/package.xml <<'EOF'
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>lerobot_robot_ur10_dg5f</name>
  <version>0.1.0</version>
  <description>LeRobot plugin and gym environment for a UR10e arm with a Tesollo DG5F hand.</description>
  <maintainer email="maintainer@example.com">DG5F workspace maintainer</maintainer>
  <license>MIT</license>

  <buildtool_depend>ament_python</buildtool_depend>

  <exec_depend>lerobot_robot_dg5f</exec_depend>

  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
EOF
```

Файл `src/lerobot_robot_ur10_dg5f/setup.py`:

```bash
# в контейнере
cat > /workspace/src/lerobot_robot_ur10_dg5f/setup.py <<'EOF'
from setuptools import find_packages, setup


package_name = "lerobot_robot_ur10_dg5f"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools", "lerobot==0.4.4"],
    zip_safe=True,
    maintainer="DG5F workspace maintainer",
    maintainer_email="maintainer@example.com",
    description="LeRobot plugin and gym environment for a UR10e arm with a DG5F hand",
    license="MIT",
)
EOF
```

Файл `src/lerobot_robot_ur10_dg5f/setup.cfg`:

```bash
# в контейнере
cat > /workspace/src/lerobot_robot_ur10_dg5f/setup.cfg <<'EOF'
[develop]
script_dir=$base/lib/lerobot_robot_ur10_dg5f

[install]
install_scripts=$base/lib/lerobot_robot_ur10_dg5f
EOF
```

`__init__.py` пока только с описанием. Окончательный импортирует модули шагов 5 и 7;
если положить его сейчас, уже тест шага 3 упадёт с `ImportError`, потому что при
импорте `lerobot_robot_ur10_dg5f.pose_math` Python сначала выполняет `__init__.py`.

```bash
# в контейнере
cat > /workspace/src/lerobot_robot_ur10_dg5f/lerobot_robot_ur10_dg5f/__init__.py <<'EOF'
"""LeRobot plugin for a UR10e arm carrying a Tesollo DG5F hand."""
EOF
```

Фикстуры, которыми пользуются тесты следующих шагов. Путь к модели и к датасету можно
переопределить переменными `UR10E_DG5F_MJCF` и `DG5F_MOCK_DATASET`.

Файл `src/lerobot_robot_ur10_dg5f/test/conftest.py`:

```bash
# в контейнере
cat > /workspace/src/lerobot_robot_ur10_dg5f/test/conftest.py <<'EOF'
import os
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def mjcf_path():
    pytest.importorskip("mujoco")
    path = Path(os.environ.get("UR10E_DG5F_MJCF", REPO_ROOT / "models" / "ur10e_dg5f" / "scene.xml"))
    if not path.is_file():
        pytest.skip(f"MJCF not found: {path}")
    return path


@pytest.fixture
def make_sim_robot(mjcf_path, tmp_path):
    from lerobot_robot_ur10_dg5f import Ur10Dg5f, Ur10Dg5fConfig

    robots = []

    def make(**overrides):
        overrides.setdefault("sim_cameras", ())
        config = Ur10Dg5fConfig(id="test", calibration_dir=tmp_path, mjcf_path=str(mjcf_path), **overrides)
        robot = Ur10Dg5f(config)
        robot.connect()
        robots.append(robot)
        return robot

    yield make
    for robot in robots:
        robot.disconnect()
EOF
```

Сборка: colcon создаёт метаданные дистрибутива, по которым LeRobot ищет плагины.
`--base-paths /workspace/src` — как в `stack.sh compile`, иначе colcon начнёт искать
пакеты по всему репо.

```bash
# в контейнере
cd /workspace
colcon build --symlink-install --base-paths /workspace/src --packages-select lerobot_robot_ur10_dg5f
source /workspace/install/setup.bash
cd /tmp && python3 -c "import importlib.metadata as m; print(m.version('lerobot_robot_ur10_dg5f'))"
```

Альтернатива без colcon — editable-установка, ссылка на исходники, а не копия:

```bash
# на хосте, вместо colcon, по желанию
pip install --user -e src/lerobot_robot_ur10_dg5f --no-deps
```

**Проверка.** Последняя команда печатает `0.1.0`: у пакета есть метаданные
дистрибутива. `cd /tmp` нужен, чтобы Python не нашёл пакет просто в текущем каталоге.
Полная проверка регистрации плагина — в шаге 5, когда появится конфиг.

### Откуда это и где найти самому

- *Правила плагинов LeRobot.* Документация «Bring Your Own Hardware»
  (<https://huggingface.co/docs/lerobot/integrate_hardware>), раздел «Using Your Own
  LeRobot Devices»: пакет должен быть **устанавливаемым**, имя начинается с
  `lerobot_robot_`, класс называется как конфиг без `Config`, оба экспортируются из
  `__init__.py`.
- *Почему одного `PYTHONPATH` мало.* Откройте
  `lerobot/utils/import_utils.py::register_third_party_plugins`: он перебирает
  `importlib.metadata.distributions()`, то есть **установленные дистрибутивы с
  метаданными**. Каталог, просто добавленный в `PYTHONPATH`, импортируется, но LeRobot
  его не найдёт. Метаданные создают `colcon build` (так собирается весь `src/`,
  `stack.sh compile`) и `pip install -e`.
- *Почему ament_python.* colcon в Humble находит пакеты по `package.xml`. Образец —
  соседний `src/lerobot_robot_dg5f/` и туториал ROS 2 «Creating a package»
  (<https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Creating-Your-First-ROS2-Package.html>).
  `setup.cfg` с `script_dir` — стандартная часть шаблона ament_python.

---

## Шаг 3. Математика поз — `pose_math.py`

**Зачем.** На этом модуле держатся residual по запястью, относительный режим перехвата
и метки для RL.

**Задание.** Функции `apply_offset(base, delta)` и `pose_difference(a, b)` (взаимно
обратные), `rotate_offset(delta, A)` (перевод смещения в другую систему) и
`rotvec_to_6d(rotvec)` (непрерывное представление ориентации для нейросети).

**Соглашение (одно на весь пакет):** поза TCP — 6 чисел, как у UR,
`[x, y, z, rx, ry, rz]`: метры и вектор поворота (axis-angle) в радианах, в системе
базы UR. Смещение задаётся в системе базы:

```latex
p = p_b + \Delta p, \qquad R = \exp([\Delta r]_\times)\, R_b
```

### Решение: команды

Файл `src/lerobot_robot_ur10_dg5f/lerobot_robot_ur10_dg5f/pose_math.py`:

```bash
# в контейнере
cat > /workspace/src/lerobot_robot_ur10_dg5f/lerobot_robot_ur10_dg5f/pose_math.py <<'EOF'
"""UR-style poses ``[x, y, z, rx, ry, rz]``: metres and a rotation vector, UR base frame."""

import numpy as np
from scipy.spatial.transform import Rotation as R


def apply_offset(base, delta):
    """Offset expressed in the UR base frame: p = p_b + dp, R = Exp(dr) @ R_b."""
    base = np.asarray(base, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    rotation = R.from_rotvec(delta[3:6]) * R.from_rotvec(base[3:6])
    return np.concatenate([base[:3] + delta[:3], rotation.as_rotvec()])


def pose_difference(a, b):
    """Inverse of apply_offset: apply_offset(b, pose_difference(a, b)) == a."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    rotation = R.from_rotvec(a[3:6]) * R.from_rotvec(b[3:6]).inv()
    return np.concatenate([a[:3] - b[:3], rotation.as_rotvec()])


def rotate_offset(delta, rotation_matrix):
    """Re-express an offset in another frame: dp -> A dp, dr -> A dr (A Exp(dr) A^T = Exp(A dr))."""
    delta = np.asarray(delta, dtype=np.float64)
    rotation_matrix = np.asarray(rotation_matrix, dtype=np.float64)
    return np.concatenate([rotation_matrix @ delta[:3], rotation_matrix @ delta[3:6]])


def rotvec_to_6d(rotvec):
    """First two columns of the rotation matrix, a continuous input for networks."""
    matrix = R.from_rotvec(np.asarray(rotvec, dtype=np.float64)).as_matrix()
    return matrix[:, :2].T.reshape(6)
EOF
```

Тесты. Обратите внимание на `same_rotation`: повороты сравниваются матрицами, потому
что около угла π вектор поворота неоднозначен.

Файл `src/lerobot_robot_ur10_dg5f/test/test_pose_math.py`:

```bash
# в контейнере
cat > /workspace/src/lerobot_robot_ur10_dg5f/test/test_pose_math.py <<'EOF'
import numpy as np
from scipy.spatial.transform import Rotation as R

from lerobot_robot_ur10_dg5f.pose_math import (
    apply_offset,
    pose_difference,
    rotate_offset,
    rotvec_to_6d,
)


def random_pose(rng):
    return np.r_[rng.normal(size=3), R.random(random_state=rng).as_rotvec()]


def same_rotation(rv_a, rv_b, atol=1e-9):
    # Compare matrices, not rotation vectors: near pi the rotation vector is ambiguous.
    return np.allclose(R.from_rotvec(rv_a).as_matrix(), R.from_rotvec(rv_b).as_matrix(), atol=atol)


def test_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(200):
        a, b = random_pose(rng), random_pose(rng)
        out = apply_offset(b, pose_difference(a, b))
        assert np.allclose(out[:3], a[:3]) and same_rotation(out[3:], a[3:])


def test_zero_offset_is_identity():
    b = random_pose(np.random.default_rng(1))
    out = apply_offset(b, np.zeros(6))
    assert np.allclose(out[:3], b[:3]) and same_rotation(out[3:], b[3:])


def test_offset_is_in_base_frame():
    base = np.r_[0.0, 0.0, 0.0, 0.0, 0.0, np.pi / 2]  # tool yawed by 90 degrees
    out = apply_offset(base, np.r_[0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert np.allclose(out[:3], [0.1, 0.0, 0.0])  # moved along base x, not tool x


def test_rotate_offset_matches_conjugation():
    rng = np.random.default_rng(2)
    a = R.random(random_state=rng).as_matrix()
    delta = random_pose(rng)
    rotated = rotate_offset(delta, a)
    expected = a @ R.from_rotvec(delta[3:]).as_matrix() @ a.T
    assert np.allclose(R.from_rotvec(rotated[3:]).as_matrix(), expected)
    assert np.allclose(rotated[:3], a @ delta[:3])


def test_6d_is_continuous_across_pi():
    just_below = rotvec_to_6d([0.0, 0.0, np.pi - 1e-6])
    just_above = rotvec_to_6d([0.0, 0.0, -(np.pi - 1e-6)])  # same rotation from the other side
    assert np.allclose(just_below, just_above, atol=1e-5)
EOF
```

Запуск. `python3 -m pytest` из каталога пакета кладёт текущий каталог в `sys.path`,
поэтому тесты импортируют пакет прямо из исходников, без пересборки:

```bash
# в контейнере
cd /workspace/src/lerobot_robot_ur10_dg5f
python3 -m pytest -q test/test_pose_math.py
```

**Проверка.** `5 passed`. Отдельно поучительно: эталонная проверка
`apply_offset(b, pose_difference(a, b)) == a` на 1000 случайных поз даёт ошибку 1.3e-15.

### Откуда это и где найти самому

- *Формат UR.* Docstring `ur_rtde`: `getActualTCPPose()` возвращает *«(x,y,z,rx,ry,rz),
  where rx, ry and rz is a rotation vector representation of the tool orientation»*.
  API: <https://sdurobotics.gitlab.io/ur_rtde/pages/reference/api.html>.
- *Композиция поворотов в scipy.* Документация `scipy.spatial.transform.Rotation`
  (<https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html>):
  `p * q` — это сначала `q`, потом `p`, то есть матрица `P @ Q`.
- *Теория ⊕ и ⊖.* Solà, Deray, Atchuthan, «A micro Lie theory for state estimation in
  robotics» (arXiv:1812.01537). Наше `apply_offset` — «левое» ⊕: возмущение в
  глобальной системе (базы). Если умножать справа (`R_b · Exp(Δr)`), смещение было бы
  в системе инструмента. Для оператора и residual естественнее система базы.
- *`rotate_offset`.* Тождество сопряжения из той же статьи: `A Exp(r) Aᵀ = Exp(A r)`
  (присоединённое представление SO(3) — сама матрица поворота). Поэтому поворотную
  часть смещения переводят в другую систему простым `A @ dr`.
- *6D-представление.* Zhou et al., «On the Continuity of Rotation Representations in
  Neural Networks», CVPR 2019 (arXiv:1812.07035): rotvec и кватернионы разрывны для
  сети, первые два столбца матрицы поворота — нет.

---

## Шаг 4. Ограничители — `safety.py`

**Зачем.** План требует «лимиты рабочей зоны» и предел поправки. Ограничитель стоит в
`Robot`, поэтому через него проходят все источники команд: политика, residual,
оператор, replay.

**Задание.** `clip_workspace(pose, lo, hi)`, `limit_step(current, target, max_lin_m,
max_ang_rad)`, `clip_hand(hand_deg, disabled)`.

### Решение: команды

Файл `src/lerobot_robot_ur10_dg5f/lerobot_robot_ur10_dg5f/safety.py`:

```bash
# в контейнере
cat > /workspace/src/lerobot_robot_ur10_dg5f/lerobot_robot_ur10_dg5f/safety.py <<'EOF'
"""Command limits applied to every action source: policy, residual, operator, replay."""

import numpy as np
from lerobot_robot_dg5f.constants import LOWER_LIMITS_DEG, UPPER_LIMITS_DEG
from lerobot_robot_dg5f.dg5f import apply_disabled_joints

from .pose_math import apply_offset, pose_difference


def clip_workspace(pose, lo_xyz, hi_xyz):
    """Clip the position to a box in the UR base frame; orientation is untouched."""
    result = np.asarray(pose, dtype=np.float64).copy()
    result[:3] = np.clip(result[:3], lo_xyz, hi_xyz)
    return result


def limit_step(current, target, max_lin_m, max_ang_rad):
    """Shorten the step current -> target to max_lin_m and max_ang_rad, keeping its direction."""
    delta = pose_difference(target, current)
    linear = np.linalg.norm(delta[:3])
    angular = np.linalg.norm(delta[3:])
    if linear > max_lin_m:
        delta[:3] *= max_lin_m / linear
    if angular > max_ang_rad:
        delta[3:] *= max_ang_rad / angular
    return apply_offset(current, delta)


def clip_hand(hand_deg, disabled_positions_deg):
    """Clamp DG5F targets to the SDK limits and hold mechanically unavailable joints."""
    hand = np.clip(np.asarray(hand_deg, dtype=np.float64), LOWER_LIMITS_DEG, UPPER_LIMITS_DEG)
    return apply_disabled_joints(hand, disabled_positions_deg)
EOF
```

Файл `src/lerobot_robot_ur10_dg5f/test/test_safety.py`:

```bash
# в контейнере
cat > /workspace/src/lerobot_robot_ur10_dg5f/test/test_safety.py <<'EOF'
import numpy as np
from lerobot_robot_dg5f.constants import BROKEN_PINKY_INDEX, UPPER_LIMITS_DEG
from scipy.spatial.transform import Rotation as R

from lerobot_robot_ur10_dg5f.safety import clip_hand, clip_workspace, limit_step


CURRENT = np.r_[0.0, 0.5, 0.5, 0.0, 0.0, 0.0]


def test_long_step_is_shortened_along_its_direction():
    target = CURRENT + np.r_[0.3, 0.0, 0.4, 0.0, 0.0, 0.0]
    out = limit_step(CURRENT, target, 0.01, 0.05)
    step = out[:3] - CURRENT[:3]
    assert np.isclose(np.linalg.norm(step), 0.01)
    assert np.allclose(step / np.linalg.norm(step), [0.6, 0.0, 0.8])


def test_short_step_is_unchanged():
    target = CURRENT + np.r_[0.002, -0.003, 0.001, 0.0, 0.0, 0.01]
    assert np.allclose(limit_step(CURRENT, target, 0.01, 0.05), target)


def test_rotation_step_is_capped():
    target = np.r_[CURRENT[:3], 1.0, 0.0, 0.0]
    out = limit_step(CURRENT, target, 0.01, 0.05)
    assert np.isclose(np.linalg.norm(R.from_rotvec(out[3:]).as_rotvec()), 0.05)
    assert np.allclose(out[3:] / np.linalg.norm(out[3:]), [1.0, 0.0, 0.0])


def test_point_outside_box_lands_on_its_face():
    out = clip_workspace(np.r_[1.0, 0.5, 0.1, 0.1, 0.2, 0.3], (-0.4, 0.35, 0.44), (0.4, 0.9, 0.8))
    assert np.allclose(out, [0.4, 0.5, 0.44, 0.1, 0.2, 0.3])


def test_hand_is_clamped_and_pinky_held():
    out = clip_hand(np.full(20, 999.0), {"rj_dg_5_1": 0.0})
    expected = UPPER_LIMITS_DEG.copy()
    expected[BROKEN_PINKY_INDEX] = 0.0
    assert np.allclose(out, expected)
EOF
```

```bash
# в контейнере
cd /workspace/src/lerobot_robot_ur10_dg5f
python3 -m pytest -q test/test_safety.py
```

**Проверка.** `5 passed`: длинный шаг укорачивается ровно до предела и сохраняет
направление `(0.6, 0, 0.8)`, короткий не меняется, поворот на 1 рад даёт ровно 0.05 рад,
точка вне коробки ложится на грань, кисть зажата, а `rj_dg_5_1` равен 0.

### Откуда это и где найти самому

- *Как это сделано в LeRobot.* `EEBoundsAndSafety` в
  `lerobot/robots/so_follower/robot_kinematic_processor.py`: сначала `np.clip` позиции по
  `end_effector_bounds`, потом проверка скачка с `max_ee_step_m = 0.05`. Но при скачке
  он **бросает исключение**. Мы вместо этого укорачиваем шаг: для RL исключение
  посреди эпизода хуже, чем медленное движение.
- *Порядок «зона, потом шаг».* Коробка выпукла, поэтому отрезок между двумя её точками
  лежит в ней. Если прошлая цель была внутри, укороченный шаг к обрезанной цели тоже
  внутри, и оба ограничения выполняются одновременно. В обратном порядке обрезка по
  зоне могла бы сама сделать скачок больше предела.
- *От чего считать шаг.* `current` — последняя **отправленная** цель, а не измеренная
  поза. Так в проекте уже устроена кисть: `src/lerobot_robot_dg5f/README.md` — «Measured
  feedback … never reseeds that command trajectory». Иначе при отставании руки
  ограничитель каждый тик разрешал бы новый полный шаг от отстающей позы.
- *Укорачивание поворота.* Масштабирование вектора относительного поворота сохраняет
  ось и уменьшает угол. Это кратчайший путь по SO(3), как slerp.
- *Кисть.* Пределы `LOWER/UPPER_LIMITS_DEG` и `apply_disabled_joints` уже есть в
  `lerobot_robot_dg5f`: импортируем, а не копируем, чтобы правда о кисти была одна.

---

## Шаг 5. Конфиг — `config_ur10_dg5f.py`

**Зачем.** Все числа ячейки в одном месте, с проверкой при создании, и регистрация
типа `ur10_dg5f` в LeRobot.

**Задание.** Dataclass-наследник `RobotConfig` с бэкендом, адресами, частотами,
пределами, рабочей зоной, home-позой и камерами.

### Решение: команды

Файл `src/lerobot_robot_ur10_dg5f/lerobot_robot_ur10_dg5f/config_ur10_dg5f.py`:

```bash
# в контейнере
cat > /workspace/src/lerobot_robot_ur10_dg5f/lerobot_robot_ur10_dg5f/config_ur10_dg5f.py <<'EOF'
"""LeRobot configuration for the UR10e + DG5F cell."""

from dataclasses import dataclass, field
from math import isfinite

from lerobot.cameras import CameraConfig
from lerobot.robots import RobotConfig
from lerobot_robot_dg5f.constants import BROKEN_PINKY_JOINT


@RobotConfig.register_subclass("ur10_dg5f")
@dataclass
class Ur10Dg5fConfig(RobotConfig):
    """Either the MuJoCo twin or the real UR10e + DG5F cell.

    Poses are in the UR base frame, as reported by ``getActualTCPPose()``.
    The workspace box below is sized for the colleague's MuJoCo table; measure
    the real one on the teach pendant before using hardware.
    """

    backend: str = "mujoco"
    mjcf_path: str = "models/ur10e_dg5f/scene.xml"
    sim_cameras: tuple[str, ...] = ("cam_front", "cam_side")
    sim_image_hw: tuple[int, int] = (240, 320)

    ur_ip: str = "192.168.1.10"
    ur_servo_hz: float = 500.0
    ur_lookahead_s: float = 0.1
    ur_gain: float = 300.0

    hand_backend: str = "tesollo"
    hand_ip: str = "169.254.186.72"
    hand_servo_hz: float = 60.0

    max_step_m: float = 0.010
    max_step_rad: float = 0.05
    workspace_lo: tuple[float, float, float] = (-0.40, 0.35, 0.44)
    workspace_hi: tuple[float, float, float] = (0.40, 0.90, 0.80)
    # initial_pose of the colleague's env raised by 10 cm: the original one is inside the table.
    home_joints_rad: tuple[float, ...] = (1.4201, -1.8540, 2.2389, -1.9449, -1.5715, -0.1499)
    hand_disabled_joints_deg: dict[str, float] = field(
        default_factory=lambda: {BROKEN_PINKY_JOINT: 0.0}
    )
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.backend not in {"mujoco", "real"}:
            raise ValueError("backend must be 'mujoco' or 'real'")
        if self.hand_backend not in {"tesollo", "mock"}:
            raise ValueError("hand_backend must be 'tesollo' or 'mock'")
        if len(self.home_joints_rad) != 6:
            raise ValueError("home_joints_rad must have 6 values")
        if len(self.workspace_lo) != 3 or len(self.workspace_hi) != 3:
            raise ValueError("workspace bounds must have 3 values each")
        if any(lo >= hi for lo, hi in zip(self.workspace_lo, self.workspace_hi)):
            raise ValueError("workspace_lo must be below workspace_hi on every axis")
        if not 0.03 <= self.ur_lookahead_s <= 0.2:
            raise ValueError("ur_lookahead_s must be in [0.03, 0.2] (servoL range)")
        if not 100.0 <= self.ur_gain <= 2000.0:
            raise ValueError("ur_gain must be in [100, 2000] (servoL range)")
        positive = {
            "ur_servo_hz": self.ur_servo_hz,
            "hand_servo_hz": self.hand_servo_hz,
            "max_step_m": self.max_step_m,
            "max_step_rad": self.max_step_rad,
        }
        for name, value in positive.items():
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
EOF
```

`__init__.py`, вторая версия: экспортирует конфиг. При импорте пакета срабатывает
`@RobotConfig.register_subclass("ur10_dg5f")` — так и происходит регистрация.

```bash
# в контейнере
cat > /workspace/src/lerobot_robot_ur10_dg5f/lerobot_robot_ur10_dg5f/__init__.py <<'EOF'
"""LeRobot plugin for a UR10e arm carrying a Tesollo DG5F hand."""

from .config_ur10_dg5f import Ur10Dg5fConfig

__all__ = ["Ur10Dg5fConfig"]
EOF
```

Файл `src/lerobot_robot_ur10_dg5f/test/test_config.py`:

```bash
# в контейнере
cat > /workspace/src/lerobot_robot_ur10_dg5f/test/test_config.py <<'EOF'
import pytest
from lerobot.robots import RobotConfig

from lerobot_robot_ur10_dg5f import Ur10Dg5fConfig


def test_defaults_are_valid_and_registered():
    Ur10Dg5fConfig()
    assert "ur10_dg5f" in RobotConfig.get_known_choices()


@pytest.mark.parametrize("overrides", [
    {"backend": "gazebo"},
    {"hand_backend": "ros"},
    {"workspace_lo": (0.5, 0.35, 0.44)},
    {"home_joints_rad": (0.0,) * 5},
    {"ur_lookahead_s": 0.5},
    {"ur_gain": 50.0},
    {"max_step_m": 0.0},
])
def test_invalid_values_are_rejected(overrides):
    with pytest.raises(ValueError):
        Ur10Dg5fConfig(**overrides)
EOF
```

Пересборка (дёшево, секунда) и две проверки: тесты конфига и то, что LeRobot находит
плагин сам, через метаданные:

```bash
# в контейнере
cd /workspace
colcon build --symlink-install --base-paths /workspace/src --packages-select lerobot_robot_ur10_dg5f
source /workspace/install/setup.bash
cd /workspace/src/lerobot_robot_ur10_dg5f
python3 -m pytest -q test/test_config.py
cd /tmp && python3 -c "
from lerobot.utils.import_utils import register_third_party_plugins
register_third_party_plugins()
from lerobot.robots import RobotConfig
print('ur10_dg5f' in RobotConfig.get_known_choices())"
```

**Проверка.** `8 passed` (значения по умолчанию валидны, тип зарегистрирован, семь
неверных значений дают `ValueError`) и `True` от проверки плагина.

### Откуда это и где найти самому

- *Базовый класс.* `lerobot/robots/config.py`: `@dataclass(kw_only=True)` с полями `id`
  и `calibration_dir`, а `__post_init__` проверяет у камер `width/height/fps`.
  `register_subclass` — из `draccus.ChoiceRegistry`. Образец в репо — `Dg5fConfig`
  (`src/lerobot_robot_dg5f/lerobot_robot_dg5f/config_dg5f.py`).
- *Диапазоны servoL.* Docstring `servoL` в `ur_rtde` 1.6.5: `lookahead_time` — «range
  [0.03,0.2]», `gain` — «range [100,2000]». Их стоит проверять в конфиге, а не узнавать
  на роботе.
- *500 Гц.* UR10e — e-Series, официальный пример ServoL из `ur_rtde` работает с
  `dt = 1.0 / 500`
  (<https://sdurobotics.gitlab.io/ur_rtde/pages/examples/high_frequency_servoing/servol_example.html>).
- *Home-поза.* `initial_pose` из `env_dg_ur.py` коллеги, поднятая через IK на 10 см.
  В исходной позе пальцы **на 8 см внутри стола**: верх стола на z = 0.150 (геом-коробка
  с полувысотой 0.075 в точке z = 0.075), кончик указательного — на z = 0.067. Пересчёт
  (после шага 6, нужен `solve_ik`):
  ```python
  model = load_model("models/ur10e_dg5f/scene.xml"); data = mujoco.MjData(model)
  data.qpos[:6] = [1.42010733, -1.74898752, 2.36328641, -2.1743184, -1.57146472, -0.14989266]
  mujoco.mj_forward(model, data)
  site = model.site("attachment_site").id
  rot = data.site_xmat[site].reshape(3, 3).copy()
  print(solve_ik(model, data, site, np.arange(6), np.arange(6), data.site_xpos[site] + [0, 0, 0.10], rot))
  ```
  Суставы руки в этой модели — первые шесть `qpos`, это видно из `model.joint(name).qposadr`.
- *Рабочая зона.* Из той же геометрии: стол в мире `x ∈ [−0.5, 0.5]`, `y ∈ [−0.95, −0.35]`,
  в системе UR — `y ∈ [0.35, 0.95]`. Кончики пальцев на 0.287 м ниже фланца (0.354 − 0.067),
  поэтому нижняя граница TCP — 0.15 + 0.287 ≈ 0.44 м. На реальном роботе зону снимают
  с пульта: подводят TCP в углы зоны задачи и читают позы.

---

## Шаг 6. Бэкенд MuJoCo — `backend.py`, `sim_backend.py`

**Зачем.** Цифровой двойник, на котором отлаживается всё остальное. План говорит, что
симуляция для эксперимента не нужна. Для разработки она нужна.

**Задание.** Общий контракт бэкенда (`RobotState`, `Backend`, фабрика `make_backend`) и
`MujocoBackend`: загрузка модели, Декартова цель → суставы (IK), кисть, шаг физики,
сброс сцены, камеры. Всё наружу — в системе базы UR.

### Решение: команды

Контракт бэкенда — без тяжёлых импортов: MuJoCo есть только в контейнере, а
`ur_rtde` — только рядом с роботом, поэтому фабрика импортирует их лениво.

Файл `src/lerobot_robot_ur10_dg5f/lerobot_robot_ur10_dg5f/backend.py`:

```bash
# в контейнере
cat > /workspace/src/lerobot_robot_ur10_dg5f/lerobot_robot_ur10_dg5f/backend.py <<'EOF'
"""Backend contract shared by the MuJoCo twin and the real cell."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np


ARM_JOINTS = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)


@dataclass
class RobotState:
    tcp: np.ndarray            # (6,) UR base frame, metres and rotation vector
    arm_q: np.ndarray          # (6,) radians
    hand_deg: np.ndarray       # (20,) degrees, JOINT_NAMES order
    protective_stop: bool = False


class Backend(Protocol):
    @property
    def is_connected(self) -> bool: ...
    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def read(self) -> RobotState: ...
    def command(self, tcp_target: np.ndarray, hand_deg: np.ndarray) -> None: ...
    def wait_next_period(self, period_s: float) -> None: ...
    def go_home(self, arm_q: np.ndarray) -> None: ...
    def images(self) -> dict[str, np.ndarray]: ...


def make_backend(config) -> Backend:
    # Lazy imports: MuJoCo lives only in the Docker image, ur_rtde only next to the robot.
    if config.backend == "mujoco":
        from .sim_backend import MujocoBackend

        return MujocoBackend(config)
    from .real_backend import RealBackend

    return RealBackend(config)
EOF
```

Файл `src/lerobot_robot_ur10_dg5f/lerobot_robot_ur10_dg5f/sim_backend.py`:

```bash
# в контейнере
cat > /workspace/src/lerobot_robot_ur10_dg5f/lerobot_robot_ur10_dg5f/sim_backend.py <<'EOF'
"""MuJoCo twin of the UR10e + DG5F cell for development without hardware."""

import mujoco
import numpy as np
from lerobot_robot_dg5f.constants import JOINT_NAMES
from scipy.spatial.transform import Rotation as R

from .backend import ARM_JOINTS, RobotState


ARM_ACTUATORS = ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")
TCP_SITE = "attachment_site"  # UR flange, tool0
# The MJCF base body is yawed by pi: the MuJoCo world is ROS base_link, while the
# UR controller reports poses in its "base" frame.
UR_BASE_IN_WORLD = R.from_euler("z", np.pi)


def load_model(mjcf_path) -> mujoco.MjModel:
    """Compile with gravity compensation on the robot, as the UR controller does.

    Setting model.body_gravcomp after compilation has no effect: MuJoCo decides
    whether to compute gravcomp from model.ngravcomp, fixed at compile time.
    """
    spec = mujoco.MjSpec.from_file(str(mjcf_path))

    def enable_gravcomp(body):
        body.gravcomp = 1.0
        for child in body.bodies:
            enable_gravcomp(child)

    enable_gravcomp(spec.body("base"))
    model = spec.compile()
    # The scene asks for RK4. With the UR actuator damping (kv=500) and armature 0.1 that is
    # unstable at 2 ms, and the saturated wrists chatter. mujoco_menagerie uses implicitfast.
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    return model


def solve_ik(model, ik_data, site_id, arm_qpos_idx, arm_dof_idx,
             target_pos_w, target_rot_w, iters=30, damping=1e-2, tol=1e-4):
    """Damped least squares: joint target that puts the site at the target pose (world frame)."""
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    for _ in range(iters):
        mujoco.mj_kinematics(model, ik_data)
        mujoco.mj_comPos(model, ik_data)  # mj_jacSite needs both
        position = ik_data.site_xpos[site_id]
        rotation = ik_data.site_xmat[site_id].reshape(3, 3)
        error = np.concatenate([
            target_pos_w - position,
            R.from_matrix(target_rot_w @ rotation.T).as_rotvec(),
        ])
        if np.linalg.norm(error) < tol:
            break
        mujoco.mj_jacSite(model, ik_data, jacp, jacr, site_id)
        jacobian = np.vstack([jacp, jacr])[:, arm_dof_idx]
        step = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + damping**2 * np.eye(6), error)
        ik_data.qpos[arm_qpos_idx] += step
    return ik_data.qpos[arm_qpos_idx].copy()


class MujocoBackend:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.data = None
        self._renderer = None

    @property
    def is_connected(self) -> bool:
        return self.model is not None

    def connect(self) -> None:
        model = load_model(self.config.mjcf_path)
        self.model = model
        self.data = mujoco.MjData(model)
        self._ik_data = mujoco.MjData(model)
        self._arm_qpos = np.array([model.joint(name).qposadr[0] for name in ARM_JOINTS])
        self._arm_dof = np.array([model.joint(name).dofadr[0] for name in ARM_JOINTS])
        self._arm_act = np.array([model.actuator(name).id for name in ARM_ACTUATORS])
        self._hand_qpos = np.array([model.joint(name).qposadr[0] for name in JOINT_NAMES])
        self._hand_act = np.array([model.actuator(f"{name}_ctrl").id for name in JOINT_NAMES])
        self._hand_lo, self._hand_hi = model.actuator_ctrlrange[self._hand_act].T
        self._site = model.site(TCP_SITE).id
        if self.config.sim_cameras:
            height, width = self.config.sim_image_hw
            self._renderer = mujoco.Renderer(model, height, width)
        self.go_home(np.asarray(self.config.home_joints_rad, dtype=np.float64))

    def disconnect(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self.model = None
        self.data = None

    def read(self) -> RobotState:
        return RobotState(
            tcp=self._tcp_in_ur_base(),
            arm_q=self.data.qpos[self._arm_qpos].copy(),
            hand_deg=np.rad2deg(self.data.qpos[self._hand_qpos]),
        )

    def command(self, tcp_target, hand_deg) -> None:
        position_w = UR_BASE_IN_WORLD.apply(tcp_target[:3])
        rotation_w = (UR_BASE_IN_WORLD * R.from_rotvec(tcp_target[3:6])).as_matrix()
        # Start IK from the current pose so it stays on the same branch (elbow up/down).
        self._ik_data.qpos[:] = self.data.qpos
        self.data.ctrl[self._arm_act] = solve_ik(
            self.model, self._ik_data, self._site, self._arm_qpos, self._arm_dof,
            position_w, rotation_w)
        self.data.ctrl[self._hand_act] = np.clip(np.deg2rad(hand_deg), self._hand_lo, self._hand_hi)

    def wait_next_period(self, period_s: float) -> None:
        substeps = max(1, round(period_s / self.model.opt.timestep))
        mujoco.mj_step(self.model, self.data, nstep=substeps)

    def go_home(self, arm_q) -> None:
        """Simulation shortcut: reset the scene and place the arm at home instantly."""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self._arm_qpos] = arm_q
        self.data.ctrl[self._arm_act] = arm_q
        self.data.ctrl[self._hand_act] = np.clip(0.0, self._hand_lo, self._hand_hi)
        mujoco.mj_forward(self.model, self.data)

    def images(self) -> dict[str, np.ndarray]:
        if self._renderer is None:
            return {}
        frames = {}
        for camera in self.config.sim_cameras:
            self._renderer.update_scene(self.data, camera=camera)
            frames[camera] = self._renderer.render().copy()
        return frames

    def _tcp_in_ur_base(self) -> np.ndarray:
        to_base = UR_BASE_IN_WORLD.inv()
        position = to_base.apply(self.data.site_xpos[self._site])
        rotation = to_base * R.from_matrix(self.data.site_xmat[self._site].reshape(3, 3))
        return np.concatenate([position, rotation.as_rotvec()])
EOF
```

Тесты бэкенда создают `Robot` из шага 7, поэтому запускаются в конце шага 7:

Файл `src/lerobot_robot_ur10_dg5f/test/test_sim_backend.py`:

```bash
# в контейнере
cat > /workspace/src/lerobot_robot_ur10_dg5f/test/test_sim_backend.py <<'EOF'
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

mujoco = pytest.importorskip("mujoco")

from lerobot_robot_ur10_dg5f.sim_backend import load_model  # noqa: E402
from lerobot_robot_ur10_dg5f.ur10_dg5f import vector_to_action  # noqa: E402


def drive(robot, target, steps):
    for _ in range(steps):
        robot.send_action(vector_to_action(target))
        robot.wait_next_period(0.1)


def angle_deg(rv_a, rv_b):
    relative = R.from_rotvec(rv_a) * R.from_rotvec(rv_b).inv()
    return np.degrees(np.linalg.norm(relative.as_rotvec()))


def test_gravity_compensation_and_stable_integrator(mjcf_path):
    model = load_model(mjcf_path)
    assert model.ngravcomp > 0
    assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST


def test_tcp_is_reported_in_ur_base_frame(make_sim_robot):
    robot = make_sim_robot()
    backend = robot.backend
    world = backend.data.site_xpos[backend._site]
    tcp = backend.read().tcp
    assert np.allclose(tcp[:3], [-world[0], -world[1], world[2]])


def test_home_is_held_without_drift(make_sim_robot):
    robot = make_sim_robot()
    start = robot.backend.read().tcp
    drive(robot, robot.last_command, 50)
    assert np.linalg.norm(robot.backend.read().tcp[:3] - start[:3]) < 1e-3


@pytest.mark.parametrize("offset", [(0.05, 0, 0), (0, 0, 0.05), (0, -0.05, 0)])
def test_tcp_reaches_5cm_offsets(make_sim_robot, offset):
    robot = make_sim_robot()
    target = robot.last_command.copy()
    target[:3] += offset
    drive(robot, target, 20)
    tcp = robot.backend.read().tcp
    assert np.linalg.norm(tcp[:3] - target[:3]) < 0.5e-3
    assert angle_deg(tcp[3:], target[3:6]) < 0.1
    wrists = robot.backend.data.actuator_force[robot.backend._arm_act][3:]
    assert np.all(np.abs(wrists) < 55.0)  # not chattering against the 56 N*m force range


def test_unreachable_target_stays_finite(make_sim_robot):
    robot = make_sim_robot()
    target = robot.last_command[:6].copy()
    target[0] += 3.0
    robot.backend.command(target, np.zeros(20))
    assert np.all(np.isfinite(robot.backend.data.ctrl))


def test_fingers_flex(make_sim_robot):
    from lerobot_robot_dg5f.constants import JOINT_NAMES

    robot = make_sim_robot()
    target = robot.last_command.copy()
    for joint in ("rj_dg_2_2", "rj_dg_2_3", "rj_dg_3_2", "rj_dg_3_3", "rj_dg_4_2", "rj_dg_4_3"):
        target[6 + JOINT_NAMES.index(joint)] = 30.0
    drive(robot, target, 20)
    hand = robot.backend.read().hand_deg
    # The simulated hand servos are soft (kp = 3 N*m/rad) and stop ~2.4 degrees short.
    assert np.max(np.abs(hand - target[6:])) < 3.0


def test_cameras_render(make_sim_robot):
    try:
        robot = make_sim_robot(sim_cameras=("cam_front",), sim_image_hw=(120, 160))
    except Exception as error:  # no OpenGL: run with MUJOCO_GL=egl
        pytest.skip(f"Rendering unavailable: {error}")
    frame = robot.get_observation()["cam_front"]
    assert frame.shape == (120, 160, 3) and frame.dtype == np.uint8 and frame.mean() > 0
EOF
```

Камера на запястье (план: сцена + запястье), по желанию. Команда вставляет её первой
строкой в тело `rl_dg_mount` и ничего не делает, если камера уже есть:

```bash
# в контейнере
python3 - <<'EOF'
from pathlib import Path
path = Path("/workspace/models/ur10e_dg5f/ur10edg5f.xml")
anchor = '<body name="rl_dg_mount" pos="0.0 0.11 0.0" quat="-1 1 -1 -1" childclass="robot">'
camera = ('<camera name="cam_wrist" pos="0.092 0.011 0.034" '
          'xyaxes="0.092 -0.993 0.074 -0.977 -0.104 -0.186" fovy="75"/>')
text = path.read_text()
if camera not in text:
    assert text.count(anchor) == 1, "anchor not found"
    path.write_text(text.replace(anchor, anchor + "\n                  " + camera))
import mujoco
model = mujoco.MjModel.from_xml_path("/workspace/models/ur10e_dg5f/scene.xml")
print([model.camera(i).name for i in range(model.ncam)])
EOF
```

Чтобы камера попала в наблюдения, допишите `"cam_wrist"` в `sim_cameras` конфига.

Эта поза вычислена, а не угадана, и проверена рендером: камера в 9 см от оси кисти со
стороны ладони смотрит вдоль пальцев за их кончики; в кадре ладонь, пальцы и стол.
Метод пригодится, когда будете ставить камеру так, как она стоит на реальном кронштейне:

```python
# home-поза, мир MuJoCo
mount = model.body("rl_dg_mount").id
p_m, R_m = data.xpos[mount], data.xmat[mount].reshape(3, 3)
tips = np.array([data.site_xpos[model.site(f"rl_dg_{i}_tip_touch").id] for i in range(1, 6)])
axis = tips.mean(0) - p_m; axis /= np.linalg.norm(axis)        # от фланца к кончикам
# сторона ладони: куда смещаются кончики, если согнуть пальцы (rj_dg_2..4_2/3 = 1 рад)
palm = ...  # смещение центра кончиков 2-4, без составляющей вдоль axis, нормированное
cam = p_m + 0.09 * palm + 0.04 * axis
z = -(tips.mean(0) + 0.15 * axis - cam); z /= np.linalg.norm(z)   # камера MuJoCo смотрит вдоль -z
x = np.cross(palm, -z); x /= np.linalg.norm(x); y = np.cross(z, x)
pos, xyaxes = R_m.T @ (cam - p_m), np.r_[R_m.T @ x, R_m.T @ y]  # в систему rl_dg_mount
```

Почему `xyaxes` и `−z`: XML reference, элемент `camera` — камера смотрит вдоль своей
отрицательной оси z, а `xyaxes` задаёт её оси x и y в системе родительского тела.

**Проверка.** Команда с камерой печатает `['cam_front', 'cam_side', 'cam_wrist']`.
Тесты `test/test_sim_backend.py` (9 штук) запускаются в конце шага 7. Числа на
эталоне при 10 Гц:

| Проверка | Результат |
| --- | --- |
| Удержание home 5 с | дрейф 0.000 мм (без gravcomp было 90 мм) |
| Шаг 5 см по x, −x, z, −y, через 2 с | ошибка 0.06–0.09 мм, ориентация < 0.005°, запястья не в насыщении |
| Цель за 3 м | `ctrl` конечный, без NaN |
| Сгибание 6 суставов пальцев до 30° | недоход до 2.4°: сервоприводы кисти в модели мягкие (`kp = 3 Н·м/рад`) |
| Кисть в середину диапазона | большой палец недоходит на 12° из-за коллизий, поэтому в тесте сгибание |

### Откуда это и где найти самому

- *Якобиан в MuJoCo.* API reference, `mj_jac`
  (<https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html>): *«The
  minimal pipeline stages required for Jacobian computations to be consistent with the
  current generalized positions mjData.qpos are mj_kinematics followed by mj_comPos»*.
  Поэтому в IK перед `mj_jacSite` стоят именно эти два вызова, на отдельной копии
  `MjData`, чтобы не трогать физику.
- *Метод IK.* Демпфированные наименьшие квадраты:

  ```latex
  \Delta q = J^\top \left(J J^\top + \lambda^2 I\right)^{-1} e, \qquad
  e = \begin{bmatrix} p^* - p \\ \log(R^* R^\top) \end{bmatrix}
  ```

  Образец в коде DeepMind: `qpos_from_site_pose` в `dm_control/utils/inverse_kinematics.py`
  (параметр `regularization_strength=3e-2`). Теория: S. Buss, «Introduction to Inverse
  Kinematics with Jacobian Transpose, Pseudoinverse and Damped Least Squares methods».
  Старт IK с текущей позы держит решение на той же ветви (локоть вверх или вниз).
- *Ловушка 1: нет компенсации гравитации.* Найдена экспериментом. TCP за 5 с уезжал на
  90 мм, а `data.actuator_force` локтя был ровно 150, то есть упирался в `forcerange`.
  Реальный контроллер UR гравитацию компенсирует, модель — нет. Атрибут
  `gravcomp` у тела: XML reference, `body/gravcomp`
  (<https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-gravcomp>).
  **Тонкость, найденная тоже экспериментом:** `model.body_gravcomp[...] = 1` после
  компиляции не меняет ничего (дрейф совпал до трёх знаков), потому что
  `model.ngravcomp` вычисляется при компиляции и остаётся 0. Работает только правка
  спецификации `MjSpec` до `compile()` (раздел Model Editing документации MuJoCo).
- *Ловушка 2: home в столе* — см. шаг 5.
- *Ловушка 3: интегратор.* В menagerie у UR10e стоит `<option integrator="implicitfast"/>`,
  и коллега его сохранил в `ur10edg5f.xml`, но `scene.xml` подключает этот файл и
  строкой ниже перезаписывает на `integrator="RK4"`. Симптом: после движения запястья
  стоят на месте, `qvel` ≈ 0.18 рад/с, а `actuator_force` = +56 при формуле актуатора
  −184. Это дрожание на частоте шага. Причина — демпфирование актуатора `kv = 500` при
  `armature = 0.1`: собственное число ≈ −kv/I ≈ −5000 1/с, `λ·dt ≈ −10` при шаге 2 мс,
  а RK4 устойчив на отрицательной полуоси только до ≈ −2.8. Документация MuJoCo,
  «Numerical integration»
  (<https://mujoco.readthedocs.io/en/stable/computation/index.html#geintegration>):
  *«The recommended integrator is implicitfast»*, неявные схемы считают такие
  скоростные силы устойчиво. С `implicitfast` шаг 5 см отрабатывается с точностью
  0.06–0.09 мм вместо 0.3–1.2 мм, ориентация — 0.005° вместо ~1°.
- *Система координат* — шаг 1. TCP — сайт `attachment_site`, это фланец UR (`tool0`).

---

## Шаг 7. Робот в контракте LeRobot — `ur10_dg5f.py`

**Зачем.** Единый интерфейс для записи демонстраций, политики и RL: тот же класс
используют `lerobot-record`, gym-среда и обёртка residual.

**Задание.** Класс `Ur10Dg5f(Robot)`: признаки наблюдений и действий, подключение,
наблюдение, отправка с ограничителями шага 4, возврат в home.

### Решение: команды

Файл `src/lerobot_robot_ur10_dg5f/lerobot_robot_ur10_dg5f/ur10_dg5f.py`:

```bash
# в контейнере
cat > /workspace/src/lerobot_robot_ur10_dg5f/lerobot_robot_ur10_dg5f/ur10_dg5f.py <<'EOF'
"""LeRobot ``Robot`` for a UR10e arm carrying a DG5F hand."""

from functools import cached_property

import numpy as np
from lerobot.processor import RobotAction, RobotObservation
from lerobot.robots import Robot
from lerobot_robot_dg5f.constants import JOINT_NAMES

from .backend import ARM_JOINTS, make_backend
from .config_ur10_dg5f import Ur10Dg5fConfig
from .safety import clip_hand, clip_workspace, limit_step


TCP_KEYS = ("tcp.x", "tcp.y", "tcp.z", "tcp.rx", "tcp.ry", "tcp.rz")
ARM_KEYS = tuple(f"{joint}.pos" for joint in ARM_JOINTS)
HAND_KEYS = tuple(f"{joint}.pos" for joint in JOINT_NAMES)
ACTION_KEYS = TCP_KEYS + HAND_KEYS


def action_to_vector(action: RobotAction) -> np.ndarray:
    missing = [key for key in ACTION_KEYS if key not in action]
    if missing:
        raise ValueError(f"Action is missing keys: {missing}")
    return np.asarray([float(action[key]) for key in ACTION_KEYS], dtype=np.float64)


def vector_to_action(vector) -> RobotAction:
    return {key: float(value) for key, value in zip(ACTION_KEYS, vector, strict=True)}


class Ur10Dg5f(Robot):
    """UR10e TCP pose (UR base frame, metres + rotation vector) and 20 DG5F joints (degrees)."""

    config_class = Ur10Dg5fConfig
    name = "ur10_dg5f"

    def __init__(self, config: Ur10Dg5fConfig):
        super().__init__(config)
        self.config = config
        self.backend = make_backend(config)
        self._last_tcp = None
        self._last_hand = None

    @cached_property
    def _camera_shapes(self) -> dict[str, tuple[int, int, int]]:
        if self.config.backend == "mujoco":
            height, width = self.config.sim_image_hw
            return {name: (height, width, 3) for name in self.config.sim_cameras}
        return {name: (cfg.height, cfg.width, 3) for name, cfg in self.config.cameras.items()}

    @cached_property
    def observation_features(self) -> dict:
        features = {key: float for key in TCP_KEYS + ARM_KEYS + HAND_KEYS}
        features.update(self._camera_shapes)
        return features

    @cached_property
    def action_features(self) -> dict:
        return {key: float for key in ACTION_KEYS}

    @property
    def is_connected(self) -> bool:
        return self.backend.is_connected

    @property
    def last_command(self) -> np.ndarray:
        """Last target actually sent: 6 TCP values followed by 20 hand degrees."""
        return np.concatenate([self._last_tcp, self._last_hand])

    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        if self.is_connected:
            return
        self.backend.connect()
        self._seed_from_measurement()

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        """UR and Tesollo report calibrated values; no LeRobot calibration needed."""

    def configure(self) -> None:
        """Nothing to configure beyond the backend connection."""

    def get_observation(self) -> RobotObservation:
        self._require_connected()
        state = self.backend.read()
        observation = dict(zip(TCP_KEYS, map(float, state.tcp)))
        observation.update(zip(ARM_KEYS, map(float, state.arm_q)))
        observation.update(zip(HAND_KEYS, map(float, state.hand_deg)))
        observation.update(self.backend.images())
        return observation

    def send_action(self, action: RobotAction) -> RobotAction:
        """Limit and send a target; return what was actually sent (record this, not the request)."""
        self._require_connected()
        requested = action_to_vector(action)
        if not np.all(np.isfinite(requested)):
            raise ValueError("UR10e + DG5F action contains NaN or infinity")
        # Box first, then step: the segment between two points of a box stays inside it,
        # so both limits hold as long as the previous target was inside the box.
        tcp = clip_workspace(requested[:6], self.config.workspace_lo, self.config.workspace_hi)
        tcp = limit_step(self._last_tcp, tcp, self.config.max_step_m, self.config.max_step_rad)
        hand = clip_hand(requested[6:], self.config.hand_disabled_joints_deg)
        self.backend.command(tcp, hand)
        self._last_tcp, self._last_hand = tcp, hand
        return vector_to_action(self.last_command)

    def go_home(self) -> None:
        self._require_connected()
        self.backend.go_home(np.asarray(self.config.home_joints_rad, dtype=np.float64))
        self._seed_from_measurement()

    def wait_next_period(self, period_s: float) -> None:
        self.backend.wait_next_period(period_s)

    def protective_stop(self) -> bool:
        return self.backend.read().protective_stop

    def disconnect(self) -> None:
        if self.is_connected:
            self.backend.disconnect()

    def _seed_from_measurement(self) -> None:
        """Seed the command trajectory from feedback once, at connect and after go_home only."""
        state = self.backend.read()
        self._last_tcp = state.tcp.copy()
        self._last_hand = clip_hand(state.hand_deg, self.config.hand_disabled_joints_deg)

    def _require_connected(self) -> None:
        if not self.is_connected:
            raise RuntimeError("UR10e + DG5F is not connected")
EOF
```

`__init__.py`, окончательная версия: экспортирует и конфиг, и робота (соглашение
LeRobot «Config/класс без Config» из шага 2):

Файл `src/lerobot_robot_ur10_dg5f/lerobot_robot_ur10_dg5f/__init__.py`:

```bash
# в контейнере
cat > /workspace/src/lerobot_robot_ur10_dg5f/lerobot_robot_ur10_dg5f/__init__.py <<'EOF'
"""LeRobot plugin for a UR10e arm carrying a Tesollo DG5F hand."""

from .config_ur10_dg5f import Ur10Dg5fConfig
from .ur10_dg5f import Ur10Dg5f

__all__ = ["Ur10Dg5f", "Ur10Dg5fConfig"]
EOF
```

Файл `src/lerobot_robot_ur10_dg5f/test/test_robot.py`:

```bash
# в контейнере
cat > /workspace/src/lerobot_robot_ur10_dg5f/test/test_robot.py <<'EOF'
import numpy as np
import pytest
from lerobot_robot_dg5f.constants import UPPER_LIMITS_DEG

from lerobot_robot_ur10_dg5f.ur10_dg5f import ACTION_KEYS, vector_to_action


def test_feature_counts(make_sim_robot):
    robot = make_sim_robot()
    assert len(robot.action_features) == 26
    assert len(robot.observation_features) == 6 + 6 + 20


def test_large_jump_is_limited_to_one_step(make_sim_robot):
    robot = make_sim_robot()
    previous = robot.last_command[:6].copy()
    target = robot.last_command.copy()
    target[1] += 0.30
    sent = robot.send_action(vector_to_action(target))
    sent_tcp = np.array([sent[key] for key in ACTION_KEYS[:6]])
    assert np.isclose(np.linalg.norm(sent_tcp[:3] - previous[:3]), robot.config.max_step_m)


def test_target_outside_workspace_is_clipped(make_sim_robot):
    robot = make_sim_robot()
    target = robot.last_command.copy()
    target[2] = 0.0  # below the table
    for _ in range(200):
        sent = robot.send_action(vector_to_action(target))
    assert np.isclose(sent["tcp.z"], robot.config.workspace_lo[2])


def test_hand_limits_and_pinky(make_sim_robot):
    robot = make_sim_robot()
    target = robot.last_command.copy()
    target[6:] = 999.0
    sent = robot.send_action(vector_to_action(target))
    hand = np.array([sent[key] for key in ACTION_KEYS[6:]])
    assert hand[16] == 0.0  # rj_dg_5_1
    assert np.allclose(np.delete(hand, 16), np.delete(UPPER_LIMITS_DEG, 16))


def test_nan_is_rejected(make_sim_robot):
    robot = make_sim_robot()
    target = robot.last_command.copy()
    target[0] = np.nan
    with pytest.raises(ValueError):
        robot.send_action(vector_to_action(target))
EOF
```

Запуск тестов этого шага и отложенных тестов бэкенда из шага 6:

```bash
# в контейнере
cd /workspace/src/lerobot_robot_ur10_dg5f
python3 -m pytest -q test/test_sim_backend.py test/test_robot.py
```

**Проверка.** `14 passed` (9 бэкенда + 5 робота): 26 действий и 32 наблюдения; прыжок
на 30 см даёт шаг ровно `max_step_m`; цель под столом упирается в `workspace_lo` по z;
кисть с 999° зажата, мизинец 0; NaN отвергается.

### Откуда это и где найти самому

- *Контракт.* `lerobot/robots/robot.py`: абстрактные `observation_features`,
  `action_features`, `is_connected`, `connect`, `is_calibrated`, `calibrate`, `configure`,
  `get_observation`, `send_action`, `disconnect`. Пошагово — та же страница «Bring Your
  Own Hardware», шаги 1–5. Две фразы оттуда определяют решение: *«these properties must
  be callable even if the robot is not yet connected»* (отсюда признаки строятся из
  конфига, а не из железа) и *«You can add safety limits (clipping, smoothing) and return
  what was actually sent»*.
- *Ключи.* Плоский словарь `"<имя>.pos"`, как у `Dg5f` и `so_follower`. Камеры —
  `{имя: (h, w, 3)}`.
- *Побочный эффект базового класса.* `Robot.__init__` создаёт каталог калибровки
  (`calibration_dir.mkdir(...)`), поэтому в тестах передаётся `calibration_dir=tmp_path`.
- *Возвращать отправленное.* Как `Dg5f.send_action`: в датасет пишется то, что ушло на
  робота после ограничителей, а не то, что просили.

---

## Перед шагом 8: что строят шаги 8–11 и по какому принципу

Синтаксис и библиотеки этого и следующих шагов разобраны построчно в [UR10_DG5F_PYTHON_PRIMER.md](UR10_DG5F_PYTHON_PRIMER.md), часть 1 — ровно `env.py`.

После шага 7 есть **робот**: ему можно дать цель и прочитать датчики. Но в дипломе
работает не робот сам по себе, а **цикл обучения из плана**: базовая политика управляет
роботом, оператор в Quest перехватывает управление при ошибке, а RL (SAC) учит маленькую
поправку к действиям политики. Награда — +1 за успех и −1 каждый раз, когда оператору
пришлось вмешаться. Шаги 8–11 строят «раму» этого цикла.

| Шаг | Что добавляет | Зачем, одной фразой |
| --- | --- | --- |
| 8, `env.py` | **время и эпизоды** вокруг робота | RL понимает только интерфейс `reset()` / `step(action)`: «начать попытку» и «сделать один такт 0.1 с» |
| 9, `synergies.py` | **сжатие кисти** с 20 суставов до 3–5 чисел | 20-мерную поправку за сотни эпизодов не выучить, 3–5 «типовых движений пальцев» — можно |
| 10, `residual.py` | **сам эксперимент**: поправка, перехват, награда | превращает 9 чисел от SAC в действие робота, подменяет его действием оператора при перехвате и считает награду |
| 11, `real_backend.py` | **настоящее железо** под тем же интерфейсом | всё, что выше бэкенда, не меняется при переходе с MuJoCo на UR10e |

### Принцип: матрёшка, каждый слой оборачивает нижний

```text
 ResidualInterventionEnv (шаг 10)   снаружи: действие = 9 чисел в [-1, 1] (поправка SAC)
   └─ Ur10Dg5fEnv (шаг 8)           снаружи: действие = 26 абсолютных чисел (поза TCP + кисть)
        └─ Ur10Dg5f, Robot (шаг 7)  снаружи: словарь {"tcp.x": ..., "rj_dg_1_1.pos": ...}
             └─ Backend (шаги 6, 11) снаружи: command(tcp, кисть), read()
```

Каждый слой делает одно дело и проверяется своими тестами. Верхний слой не знает, что
под ним — MuJoCo или реальный UR10e: поэтому всю логику можно отладить в симуляторе,
а шаг 11 меняет только самый нижний слой. Это тот же приём, что `gym.Wrapper` в
Gymnasium: обёртка меняет вид действий для внешнего мира, но внутри зовёт ту же
`step()`.

### Один такт целиком (0.1 с)

```text
 SAC ── Δa: 9 чисел ──────┐
 оператор: перехват? ─────┤  ResidualInterventionEnv (шаг 10)
 базовая политика: a_base ┘    нет перехвата: a = a_base ⊕ поправка (Δa × масштабы)
                               перехват:      a = поза руки оператора относительно момента
                                              нажатия, пальцы плавно переходят к оператору
                                       │
                                       ▼
                     Ur10Dg5fEnv.step(a) (шаг 8):
                     Robot.send_action(a) → ограничители → бэкенд → 0.1 с физики → наблюдение
                                       │
                                       ▼
     награда: +1 за успех, −1 в такт, когда начался перехват, иначе 0
     info: a_base, что реально исполнено, метка-поправка, флаг «это шаг оператора»
     → это и есть данные для двух буферов SAC
```

Пример с числами. SAC выдал `Δa = [0.5, 0, 0, 0, 0, 0, 0.2, 0, 0]`. Масштабы
по умолчанию: 2 см, 0.1 рад, 10 единиц синергии. Значит, к позе базовой политики
добавится 1 см по x и 2 единицы первой синергии кисти (например, «чуть сжать все
пальцы»). Если в этот такт оператор нажал перехват, то Δa игнорируется, рука идёт за
рукой оператора, награда −1, а в `info` записывается, какой поправкой SAC *мог бы*
получить действие оператора. Так SAC учится и на своих шагах, и на шагах человека.

### Чего ещё нет и чем это заменено в тестах

Шаги 8–10 строят раму, в которую потом вставляются настоящие части. Пока их нет,
тесты подставляют заглушки с тем же интерфейсом:

| Часть цикла | Сейчас, в тестах | Потом, по плану |
| --- | --- | --- |
| Базовая политика | `hold_policy`: «стой, где стоишь» | Diffusion Policy, обученная на ~50 демонстрациях |
| Оператор | `ScriptedOperator`: сценарий «нажал на такте 10, отпустил на 30» | плагин телеоператора Quest |
| SAC | фиксированные или случайные Δa | learner HIL-SERL из LeRobot |
| Синергии | синтетические позы | PCA по вашим демонстрациям |
| Робот | MuJoCo | UR10e и DG-5F (шаг 11) |

### Два термина из Gymnasium, без которых шаг 8 непонятен

- `reset()` начинает попытку: робот в home, сцена сброшена. Возвращает первое наблюдение.
- `step(action)` делает один такт и возвращает пять вещей: наблюдение, награду,
  `terminated` (попытка закончилась сама: успех или защитная остановка), `truncated`
  (кончилось время, 30 с) и `info` (всё остальное для логов).

---

## Шаг 8. Gym-среда — `env.py`

**Зачем.** Стандартный интерфейс `reset/step` для RL и для обёртки шага 10.

**Задание.** `Ur10Dg5fEnv(gym.Env)`: пространства в ключах LeRobot, reset в home с
ожиданием сброса сцены, `step` с темпом `fps`, `terminated` при защитной остановке,
`truncated` по времени, отправленное действие в `info`.

### Решение: команды

Файл `src/lerobot_robot_ur10_dg5f/lerobot_robot_ur10_dg5f/env.py`:

```bash
# в контейнере
cat > /workspace/src/lerobot_robot_ur10_dg5f/lerobot_robot_ur10_dg5f/env.py <<'EOF'
"""Gymnasium environment around the UR10e + DG5F LeRobot robot."""

from typing import Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

from .pose_math import rotvec_to_6d
from .ur10_dg5f import ACTION_KEYS, ARM_KEYS, HAND_KEYS, TCP_KEYS, Ur10Dg5f, vector_to_action


# TCP position (3) + 6D orientation (6) + arm joints (6) + hand joints (20).
STATE_DIM = 3 + 6 + len(ARM_KEYS) + len(HAND_KEYS)


class Ur10Dg5fEnv(gym.Env):
    """Absolute 26-value actions: TCP pose (6) and DG5F joints in degrees (20).

    Limits live in the robot, not in the action space, so replay and any policy
    go through the same safety path. Reward is zero here; the residual wrapper adds it.
    """

    metadata = {"render_modes": []}

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
        self.action_space = spaces.Box(-np.inf, np.inf, shape=(len(ACTION_KEYS),), dtype=np.float32)
        observation_spaces = {
            OBS_STATE: spaces.Box(-np.inf, np.inf, shape=(STATE_DIM,), dtype=np.float32),
        }
        for name, shape in robot.observation_features.items():
            if isinstance(shape, tuple):
                observation_spaces[f"{OBS_IMAGES}.{name}"] = spaces.Box(0, 255, shape=shape, dtype=np.uint8)
        self.observation_space = spaces.Dict(observation_spaces)
        self._step_count = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.robot.go_home()
        if self.wait_for_scene_reset is not None:
            self.wait_for_scene_reset()  # real cell: operator puts the object back, presses a key
        self._step_count = 0
        return self._observation(), {}

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

    def close(self):
        self.robot.disconnect()

    def _observation(self) -> dict[str, np.ndarray]:
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
EOF
```

Файл `src/lerobot_robot_ur10_dg5f/test/test_env.py`:

```bash
# в контейнере
cat > /workspace/src/lerobot_robot_ur10_dg5f/test/test_env.py <<'EOF'
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from lerobot.utils.constants import OBS_STATE

from lerobot_robot_ur10_dg5f.env import STATE_DIM, Ur10Dg5fEnv


def test_passes_gymnasium_checker(make_sim_robot):
    env = Ur10Dg5fEnv(make_sim_robot(), fps=10.0, episode_s=2.0)
    check_env(env, skip_render_check=True)


def test_image_keys_follow_lerobot_naming(make_sim_robot):
    try:
        robot = make_sim_robot(sim_cameras=("cam_front",), sim_image_hw=(60, 80))
    except Exception as error:
        pytest.skip(f"Rendering unavailable: {error}")
    observation, _ = Ur10Dg5fEnv(robot).reset()
    assert set(observation) == {OBS_STATE, "observation.images.cam_front"}
    assert observation[OBS_STATE].shape == (STATE_DIM,)


def test_hold_action_keeps_tcp(make_sim_robot):
    env = Ur10Dg5fEnv(make_sim_robot())
    env.reset()
    hold = env.robot.last_command.copy()
    start = env.robot.backend.read().tcp[:3]
    for _ in range(100):
        env.step(hold)
    assert np.linalg.norm(env.robot.backend.read().tcp[:3] - start) < 1e-3


def test_truncates_exactly_at_episode_length(make_sim_robot):
    env = Ur10Dg5fEnv(make_sim_robot(), fps=10.0, episode_s=1.5)
    env.reset()
    hold = env.robot.last_command.copy()
    flags = [env.step(hold)[3] for _ in range(15)]
    assert flags == [False] * 14 + [True]
EOF
```

```bash
# в контейнере
cd /workspace/src/lerobot_robot_ur10_dg5f
python3 -m pytest -q test/test_env.py
```

**Проверка.** `4 passed`, плюс пять предупреждений `check_env` про бесконечные границы
Box — ожидаемы. Тесты: `check_env` проходит; ключи изображений `observation.images.<cam>`;
100 шагов «стой» не двигают TCP больше чем на 1 мм; `truncated` ровно на 15-м шаге при
`episode_s = 1.5`, `fps = 10`.

### Откуда это и где найти самому

- *Своя среда.* Gymnasium, «Create a Custom Environment»
  (<https://gymnasium.farama.org/introduction/create_custom_env/>): про
  `super().reset(seed=seed)` там сказано *«IMPORTANT: Must call this first to seed the
  random number generator»*; там же `check_env` из `gymnasium.utils.env_checker`.
- *Имена ключей.* `lerobot/utils/constants.py`: `OBS_STATE = "observation.state"`,
  `OBS_IMAGES = "observation.images"`. С ними датасет и политики LeRobot примут
  наблюдения без переименования.
- *Шаблон и почему он не годится.* `RobotEnv` в `lerobot/rl/gym_manipulator.py` —
  хороший образец структуры, но завязан на `robot.bus.motors`.
- *Пределы в Robot, а не в action_space.* Тогда replay и любая политика идут через одни
  ограничители. `check_env` предупреждает про бесконечные границы Box. Это ожидаемо:
  нормированное пространство `[-1, 1]` появится в обёртке шага 10.
- *Ориентация в state — 6D*, см. Zhou et al. на шаге 3. `tcp.*` в данных Robot остаются
  rotvec, как у UR.

---

## Шаг 9. Синергии кисти — `synergies.py`

**Зачем.** План: поправка по кисти живёт в 3–5 синергиях, а не в 20 суставах.

**Задание.** `HandSynergies` с `fit` (PCA), переводом смещений между суставами и
синергиями, сохранением в `npz` и загрузкой поз кисти из датасета LeRobot.

### Решение: команды

Файл `src/lerobot_robot_ur10_dg5f/lerobot_robot_ur10_dg5f/synergies.py`:

```bash
# в контейнере
cat > /workspace/src/lerobot_robot_ur10_dg5f/lerobot_robot_ur10_dg5f/synergies.py <<'EOF'
"""Hand synergies: PCA over demonstrated DG5F postures."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from lerobot_robot_dg5f.constants import JOINT_NAMES


@dataclass
class HandSynergies:
    mean_deg: np.ndarray                  # (20,)
    components: np.ndarray                # (k, 20), orthonormal rows
    explained_variance_ratio: np.ndarray  # (k,)

    @property
    def k(self) -> int:
        return self.components.shape[0]

    @classmethod
    def fit(cls, hand_deg, k: int) -> "HandSynergies":
        data = np.asarray(hand_deg, dtype=np.float64)
        if data.ndim != 2 or data.shape[1] != len(JOINT_NAMES):
            raise ValueError(f"Expected an (N, {len(JOINT_NAMES)}) array of hand postures")
        if not 1 <= k <= min(data.shape):
            raise ValueError("k must be between 1 and min(N, 20)")
        mean = data.mean(axis=0)
        _, singular, vt = np.linalg.svd(data - mean, full_matrices=False)
        variance = singular**2
        total = variance.sum() or 1.0
        return cls(mean, vt[:k].copy(), variance[:k] / total)

    def to_joint_offset(self, dz) -> np.ndarray:
        """Synergy coordinates (k,) -> joint offset in degrees (20,)."""
        return self.components.T @ np.asarray(dz, dtype=np.float64)

    def project_offset(self, dq) -> np.ndarray:
        """Joint offset in degrees (20,) -> synergy coordinates (k,)."""
        return self.components @ np.asarray(dq, dtype=np.float64)

    def save(self, path) -> None:
        np.savez(path, mean_deg=self.mean_deg, components=self.components,
                 explained_variance_ratio=self.explained_variance_ratio)

    @classmethod
    def load(cls, path) -> "HandSynergies":
        with np.load(path) as data:
            return cls(data["mean_deg"], data["components"], data["explained_variance_ratio"])


def hand_actions_from_dataset(root) -> np.ndarray:
    """DG5F joint targets (N, 20) from the action column of a local LeRobot v3 dataset."""
    import pandas as pd

    root = Path(root)
    names = json.loads((root / "meta" / "info.json").read_text())["features"]["action"]["names"]
    # The DG5F recorder names actions "rj_dg_1_1"; the Robot API names them "rj_dg_1_1.pos".
    index = {name.removesuffix(".pos"): i for i, name in enumerate(names)}
    columns = [index[joint] for joint in JOINT_NAMES]
    files = sorted((root / "data").rglob("*.parquet"))
    actions = np.stack(pd.concat(pd.read_parquet(f, columns=["action"]) for f in files)["action"].to_numpy())
    return actions[:, columns]
EOF
```

Файл `src/lerobot_robot_ur10_dg5f/test/test_synergies.py`:

```bash
# в контейнере
cat > /workspace/src/lerobot_robot_ur10_dg5f/test/test_synergies.py <<'EOF'
import os

import numpy as np
import pytest
from lerobot_robot_dg5f.constants import BROKEN_PINKY_INDEX

from conftest import REPO_ROOT
from lerobot_robot_ur10_dg5f.synergies import HandSynergies, hand_actions_from_dataset


def synthetic_postures(rng, n=500):
    """Postures from three known directions plus noise; the pinky base stays at zero."""
    basis = np.linalg.qr(rng.normal(size=(20, 3)))[0].T
    basis[:, BROKEN_PINKY_INDEX] = 0.0
    basis = np.linalg.qr(basis.T)[0].T
    weights = rng.normal(scale=[30.0, 15.0, 8.0], size=(n, 3))
    data = 10.0 + weights @ basis + rng.normal(scale=0.3, size=(n, 20))
    data[:, BROKEN_PINKY_INDEX] = 0.0
    return data, basis


def test_recovers_known_subspace():
    data, basis = synthetic_postures(np.random.default_rng(0))
    synergies = HandSynergies.fit(data, k=3)
    # Singular values of the overlap matrix are the cosines of the principal angles.
    cosines = np.linalg.svd(synergies.components @ basis.T, compute_uv=False)
    assert np.all(cosines > 0.999)
    assert synergies.explained_variance_ratio.sum() > 0.99


def test_offsets_roundtrip_and_pinky_has_no_weight():
    data, _ = synthetic_postures(np.random.default_rng(1))
    synergies = HandSynergies.fit(data, k=3)
    dz = np.array([1.0, -2.0, 0.5])
    assert np.allclose(synergies.project_offset(synergies.to_joint_offset(dz)), dz)
    assert np.allclose(synergies.components[:, BROKEN_PINKY_INDEX], 0.0, atol=1e-9)


def test_save_and_load(tmp_path):
    data, _ = synthetic_postures(np.random.default_rng(2))
    synergies = HandSynergies.fit(data, k=4)
    synergies.save(tmp_path / "synergies.npz")
    loaded = HandSynergies.load(tmp_path / "synergies.npz")
    assert np.allclose(loaded.components, synergies.components)
    assert loaded.k == 4


def test_reads_hand_actions_from_recorded_dataset():
    root = os.environ.get("DG5F_MOCK_DATASET", REPO_ROOT / "lerobot_datasets" / "mock_dataset_and_debug_20260908")
    if not (os.path.isdir(root)):
        pytest.skip(f"Dataset not found: {root}")
    actions = hand_actions_from_dataset(root)
    assert actions.ndim == 2 and actions.shape[1] == 20 and len(actions) > 0
    assert np.all(np.isfinite(actions))
EOF
```

```bash
# в контейнере
cd /workspace/src/lerobot_robot_ur10_dg5f
python3 -m pytest -q test/test_synergies.py
```

**Проверка.** `4 passed`: подпространство из трёх известных направлений
восстанавливается (косинусы главных углов > 0.999); прямой и обратный переходы
согласованы, у `rj_dg_5_1` нулевой вес; сохранение и загрузка; из мок-датасета читается
массив `(N, 20)`. Когда появятся демонстрации:
`HandSynergies.fit(hand_actions_from_dataset(root), k=5)`, и смотрите на
`explained_variance_ratio`.

### Откуда это и где найти самому

- *Идея синергий.* Santello, Flanders, Soechting, «Postural hand synergies for tool use»,
  J. Neuroscience, 1998: несколько главных компонент объясняют большую часть дисперсии
  поз человеческой руки. Для роботизированных кистей — Ciocarlie, Allen, «Hand Posture
  Subspaces for Dexterous Robotic Grasping», IJRR 2009 («eigengrasps»).
- *PCA через SVD.* Для центрированных данных `X − μ = U S Vᵀ` строки `Vᵀ` —
  ортонормированные главные оси, доля дисперсии — `s² / Σs²`. Ортонормированность даёт
  `project(to_joint(dz)) = dz`, а переход назад — простое транспонирование.
- *Формат датасета.* Посмотрите на свой же датасет:
  `lerobot_datasets/mock_dataset_and_debug_20260908/meta/info.json` — формат v3.0,
  `features.action.names` = `["rj_dg_1_1", …]` (без `.pos`, так пишет рекордер кисти),
  данные — `data/chunk-*/file-*.parquet`, колонка `action`. Читать проще прямо через
  pandas: не нужно декодировать видео.

---

## Шаг 10. Residual и перехват — `residual.py`

**Зачем.** Это ядро плана: исполняемое действие, относительный режим руки, плавный
переход пальцев, награда RLIF и метки для двух буферов SAC.

**Задание.** Обёртка `ResidualInterventionEnv(gym.Wrapper)` над `Ur10Dg5fEnv` и чистые
функции `apply_residual`, `residual_label`, `follow_operator`.

```latex
a_t = \begin{cases} a^{op}_t, & \text{оператор держит перехват} \\
a^{base}_t \oplus \mathrm{scale}_\beta\!\left(\mathrm{clip}(\Delta a_t, -1, 1)\right), & \text{иначе} \end{cases}
\qquad
r_t = \mathbb{1}[\text{успех}_t] - \mathbb{1}[\text{вмешательство началось на шаге } t]
```

Относительный режим (`R_al` — поворот из системы Quest в систему базы UR, калибруется
один раз):

```latex
p^{*} = p_{rob,0} + R_{al}\,(p_{op} - p_{op,0}), \qquad
R^{*} = \left(R_{al} R_{op} R_{op,0}^\top R_{al}^\top\right) R_{rob,0}
```

### Решение: команды

Файл `src/lerobot_robot_ur10_dg5f/lerobot_robot_ur10_dg5f/residual.py`:

```bash
# в контейнере
cat > /workspace/src/lerobot_robot_ur10_dg5f/lerobot_robot_ur10_dg5f/residual.py <<'EOF'
"""Residual RL over a frozen base policy with operator takeover and the RLIF reward."""

from dataclasses import dataclass
from typing import Callable, Protocol

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from lerobot.teleoperators.utils import TeleopEvents

from .pose_math import apply_offset, pose_difference, rotate_offset
from .synergies import HandSynergies


@dataclass(frozen=True)
class ResidualScale:
    """beta of the plan, one per unit: metres, radians, synergy units (degrees along a component)."""

    lin_m: float = 0.02
    ang_rad: float = 0.10
    syn: float = 10.0

    def wrist(self) -> np.ndarray:
        return np.array([self.lin_m] * 3 + [self.ang_rad] * 3)


class OperatorSource(Protocol):
    def get_action(self) -> dict:
        """{"wrist_pose": (6,) in the Quest frame, "hand_deg": (20,) retargeted DG5F joints}."""

    def get_teleop_events(self) -> dict:
        """Keys from lerobot.teleoperators.utils.TeleopEvents."""


def apply_residual(base, residual, synergies: HandSynergies, scale: ResidualScale) -> np.ndarray:
    """a = a_base (+) scale * clip(residual): wrist offset in the base frame, hand via synergies."""
    residual = np.clip(np.asarray(residual, dtype=np.float64), -1.0, 1.0)
    tcp = apply_offset(base[:6], residual[:6] * scale.wrist())
    hand = base[6:] + synergies.to_joint_offset(residual[6:] * scale.syn)
    return np.concatenate([tcp, hand])


def residual_label(executed, base, synergies: HandSynergies, scale: ResidualScale):
    """Normalised residual that would turn base into executed; clipped, plus a saturation flag."""
    raw = np.concatenate([
        pose_difference(executed[:6], base[:6]),
        synergies.project_offset(np.asarray(executed[6:]) - np.asarray(base[6:])),
    ])
    scales = np.concatenate([scale.wrist(), np.full(synergies.k, scale.syn)])
    normalised = np.divide(raw, scales, out=np.zeros_like(raw), where=scales > 0)
    saturated = bool(np.any(np.abs(normalised) > 1.0) or np.any((scales == 0) & (np.abs(raw) > 1e-9)))
    return np.clip(normalised, -1.0, 1.0), saturated


def follow_operator(operator_pose, operator_anchor, robot_anchor, quest_to_base) -> np.ndarray:
    """Relative mode: move the robot from its anchor by the operator's motion since takeover."""
    delta_quest = pose_difference(operator_pose, operator_anchor)
    return apply_offset(robot_anchor, rotate_offset(delta_quest, quest_to_base))


class ResidualInterventionEnv(gym.Wrapper):
    """Action: residual in [-1, 1]^(6+k). Reward: 1[success] - 1[takeover started this step]."""

    def __init__(
        self,
        env: gym.Env,
        base_policy: Callable[[dict], np.ndarray],
        synergies: HandSynergies,
        operator: OperatorSource,
        scale: ResidualScale = ResidualScale(),
        quest_to_base=np.eye(3),
        blend_s: float = 0.4,
    ):
        super().__init__(env)
        self.base_policy = base_policy
        self.synergies = synergies
        self.operator = operator
        self.scale = scale
        self.quest_to_base = np.asarray(quest_to_base, dtype=np.float64)
        self.blend_steps = max(1, round(blend_s / env.unwrapped.period_s))
        self.action_space = spaces.Box(-1.0, 1.0, shape=(6 + synergies.k,), dtype=np.float32)

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._observation = observation
        self._intervening = False
        self._last_executed = self.env.unwrapped.robot.last_command.copy()
        self._operator_anchor = None
        self._robot_anchor = None
        self._blend_from = None
        self._blend_step = 0
        return observation, info

    def step(self, residual):
        events = self.operator.get_teleop_events()
        intervening = bool(events.get(TeleopEvents.IS_INTERVENTION, False))
        started = intervening and not self._intervening
        released = self._intervening and not intervening
        # The base policy runs during takeover too: labels need a_base.
        base = np.asarray(self.base_policy(self._observation), dtype=np.float64)

        if started or released:
            self._blend_from = self._last_executed[6:].copy()
            self._blend_step = 0

        if intervening:
            operator = self.operator.get_action()
            operator_pose = np.asarray(operator["wrist_pose"], dtype=np.float64)
            if started:
                self._operator_anchor = operator_pose.copy()
                self._robot_anchor = self._last_executed[:6].copy()
            tcp = follow_operator(operator_pose, self._operator_anchor, self._robot_anchor,
                                  self.quest_to_base)
            hand_goal = np.asarray(operator["hand_deg"], dtype=np.float64)
            applied = None
        else:
            applied = np.clip(np.asarray(residual, dtype=np.float64), -1.0, 1.0)
            proposal = apply_residual(base, applied, self.synergies, self.scale)
            tcp, hand_goal = proposal[:6], proposal[6:]

        hand = self._blend(hand_goal)
        observation, reward, terminated, truncated, info = self.env.step(np.concatenate([tcp, hand]))
        executed = np.asarray(info["sent_action"], dtype=np.float64)

        if intervening:
            label, saturated = residual_label(executed, base, self.synergies, self.scale)
        else:
            label, saturated = applied, False

        success = bool(events.get(TeleopEvents.SUCCESS, False))
        reward = float(reward) + float(success) - float(started)
        terminated = bool(terminated or success or events.get(TeleopEvents.TERMINATE_EPISODE, False))
        info.update(
            base_action=base,
            executed_action=executed,
            residual_label=label,
            label_saturated=saturated,
            is_intervention=intervening,
            intervention_started=started,
            success=success,
        )
        self._observation = observation
        self._intervening = intervening
        self._last_executed = executed
        return observation, reward, terminated, truncated, info

    def _blend(self, goal):
        """Linear finger transition over blend_s after takeover and after release."""
        if self._blend_from is None:
            return goal
        self._blend_step += 1
        alpha = min(1.0, self._blend_step / self.blend_steps)
        blended = (1.0 - alpha) * self._blend_from + alpha * goal
        if alpha >= 1.0:
            self._blend_from = None
        return blended
EOF
```

Тесты: подделка среды `FakeEnv` исполняет ровно то, что ей дали, поэтому логика обёртки
проверяется отдельно от физики, а последний тест — на настоящей MuJoCo-среде.

Файл `src/lerobot_robot_ur10_dg5f/test/test_residual.py`:

```bash
# в контейнере
cat > /workspace/src/lerobot_robot_ur10_dg5f/test/test_residual.py <<'EOF'
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from lerobot.teleoperators.utils import TeleopEvents
from scipy.spatial.transform import Rotation as R

from lerobot_robot_ur10_dg5f.residual import (
    ResidualInterventionEnv,
    ResidualScale,
    apply_residual,
    residual_label,
)
from lerobot_robot_ur10_dg5f.synergies import HandSynergies


HOME = np.r_[-0.10, 0.50, 0.45, 2.2, -2.2, 0.0, np.zeros(20)]


def make_synergies(k=3):
    components = np.linalg.qr(np.random.default_rng(0).normal(size=(20, k)))[0].T
    return HandSynergies(np.zeros(20), components, np.full(k, 1.0 / k))


class FakeRobot:
    def __init__(self):
        self.last_command = HOME.copy()


class FakeEnv(gym.Env):
    """Executes exactly what it is given, so the wrapper is tested in isolation."""

    period_s = 0.1
    observation_space = spaces.Box(-np.inf, np.inf, shape=(26,))
    action_space = spaces.Box(-np.inf, np.inf, shape=(26,))

    def __init__(self):
        self.robot = FakeRobot()
        self.sent = []

    def reset(self, *, seed=None, options=None):
        self.robot.last_command = HOME.copy()
        return HOME.copy(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        self.sent.append(action)
        self.robot.last_command = action.copy()
        return action.copy(), 0.0, False, False, {"sent_action": action.copy()}


class ScriptedOperator:
    def __init__(self, takeover_steps=(), success_step=None, wrist=None, hand=None):
        self.takeover_steps = set(takeover_steps)
        self.success_step = success_step
        self.step = -1
        self.wrist = np.r_[5.0, 5.0, 5.0, 0.0, 0.0, 0.0] if wrist is None else wrist
        self.hand = np.full(20, 40.0) if hand is None else hand

    def get_teleop_events(self):
        self.step += 1
        return {
            TeleopEvents.IS_INTERVENTION: self.step in self.takeover_steps,
            TeleopEvents.SUCCESS: self.step == self.success_step,
        }

    def get_action(self):
        return {"wrist_pose": self.wrist.copy(), "hand_deg": self.hand.copy()}


def hold_policy(observation):
    return HOME.copy()


def same_pose(a, b, atol=1e-9):
    return (np.allclose(a[:3], b[:3], atol=atol)
            and np.allclose(R.from_rotvec(a[3:6]).as_matrix(), R.from_rotvec(b[3:6]).as_matrix(), atol=atol)
            and np.allclose(a[6:], b[6:], atol=atol))


def make_env(operator, scale=ResidualScale(), policy=hold_policy):
    env = ResidualInterventionEnv(FakeEnv(), policy, make_synergies(), operator, scale=scale)
    env.reset()
    return env


def test_beta_zero_reproduces_base_policy():
    rng = np.random.default_rng(3)
    env = make_env(ScriptedOperator(), scale=ResidualScale(0.0, 0.0, 0.0))
    for _ in range(50):
        _, _, _, _, info = env.step(rng.uniform(-1, 1, size=9))
        assert same_pose(info["executed_action"], info["base_action"])


def test_autonomous_label_is_the_clipped_residual():
    env = make_env(ScriptedOperator())
    residual = np.r_[2.0, -0.5, 0.1, 0.0, 0.0, 0.3, -3.0, 0.2, 0.0]
    _, reward, _, _, info = env.step(residual)
    assert np.allclose(info["residual_label"], np.clip(residual, -1, 1))
    assert reward == 0.0 and not info["is_intervention"]


def test_rlif_penalty_only_on_takeover_start():
    env = make_env(ScriptedOperator(takeover_steps=range(10, 30)))
    rewards, flags, starts = [], [], []
    for _ in range(40):
        _, reward, _, _, info = env.step(np.zeros(9))
        rewards.append(reward)
        flags.append(info["is_intervention"])
        starts.append(info["intervention_started"])
    assert [i for i, r in enumerate(rewards) if r != 0] == [10]
    assert rewards[10] == -1.0
    assert [i for i, f in enumerate(flags) if f] == list(range(10, 30))
    assert [i for i, s in enumerate(starts) if s] == [10]


def test_takeover_does_not_jump_the_arm():
    env = make_env(ScriptedOperator(takeover_steps=range(3, 10)))  # operator hand is 8 m away
    for _ in range(3):
        env.step(np.full(9, 0.5))
    before = env.env.robot.last_command.copy()
    _, _, _, _, info = env.step(np.zeros(9))
    assert info["intervention_started"]
    assert np.allclose(info["executed_action"][:3], before[:3])


def test_fingers_blend_in_and_back_out():
    operator = ScriptedOperator(takeover_steps=range(0, 10), hand=np.full(20, 40.0))
    env = make_env(operator)
    hands = [env.step(np.zeros(9))[4]["executed_action"][6:].mean() for _ in range(20)]
    assert np.isclose(hands[0], 10.0)  # 0.4 s blend at 10 Hz: a quarter of the way on the first step
    assert np.all(np.diff(hands[:4]) > 0) and np.isclose(hands[3], 40.0)
    assert np.isclose(hands[10], 30.0) and np.isclose(hands[13], 0.0)  # and back to the policy


def test_operator_label_recovers_small_corrections():
    synergies = make_synergies()
    scale = ResidualScale()
    base = HOME.copy()
    true_residual = np.r_[0.3, -0.2, 0.1, 0.05, -0.4, 0.2, 0.5, -0.7, 0.1]
    executed = apply_residual(base, true_residual, synergies, scale)
    label, saturated = residual_label(executed, base, synergies, scale)
    assert np.allclose(label, true_residual) and not saturated


def test_large_operator_correction_saturates():
    synergies = make_synergies()
    executed = HOME.copy()
    executed[0] += 0.10  # 10 cm against lin_m = 2 cm
    label, saturated = residual_label(executed, HOME, synergies, ResidualScale())
    assert saturated and label[0] == 1.0


def test_success_terminates_with_reward():
    env = make_env(ScriptedOperator(success_step=5))
    for step in range(6):
        _, reward, terminated, _, _ = env.step(np.zeros(9))
    assert reward == 1.0 and terminated


def test_wraps_the_simulated_cell(make_sim_robot):
    from lerobot_robot_ur10_dg5f.env import Ur10Dg5fEnv

    env = Ur10Dg5fEnv(make_sim_robot())
    wrapper = ResidualInterventionEnv(
        env, lambda observation: env.robot.last_command.copy(), make_synergies(),
        ScriptedOperator(), scale=ResidualScale(0.0, 0.0, 0.0))
    wrapper.reset()
    for _ in range(10):
        _, _, _, _, info = wrapper.step(np.ones(9))
        assert same_pose(info["executed_action"], info["base_action"], atol=1e-9)
EOF
```

```bash
# в контейнере
cd /workspace/src/lerobot_robot_ur10_dg5f
python3 -m pytest -q test/test_residual.py
```

**Проверка.** `9 passed`, включая проверку, которую план требует явно: **при β = 0
исполняемое совпадает с базовым** при любом Δa (и на подделке, и в MuJoCo). Ещё: −1
только на шаге начала перехвата; при нажатии рука не прыгает, даже если рука оператора
в 8 м; пальцы переходят за 0.4 с (10 → 20 → 30 → 40° при 10 Гц) и так же обратно;
метка восстанавливает малую поправку и насыщается на большой; успех даёт +1 и
`terminated`.

### Откуда это и где найти самому

- *План*, разделы «Вмешательства и награда» и «Алгоритмы и готовый код» — формулы
  выше оттуда.
- *Награда.* RLIF (Luo et al., ICLR 2024): штраф −1 именно в момент начала
  вмешательства, а не за каждый шаг удержания.
- *Флаги и буферы.* HIL-SERL: шаги оператора и робота хранятся раздельно. В LeRobot
  посмотрите `InterventionActionProcessorStep` в `lerobot/processor/hil_processor.py`:
  при `TeleopEvents.IS_INTERVENTION` он подменяет действие политики на `teleop_action`.
  Сами события — перечисление `TeleopEvents` в `lerobot/teleoperators/utils.py`
  (`IS_INTERVENTION`, `SUCCESS`, `TERMINATE_EPISODE`, `RERECORD_EPISODE`). Те же ключи —
  значит, будущий телеоператор Quest подключится без переделки.
- *Residual поверх замороженной политики.* ResFiT (Ankile et al., 2025): поправка
  ограничена масштабом. У плана один β, но единицы разные (метры, радианы, синергии),
  поэтому масштабов три.
- *Плавный переход.* BORA и Hand-in-the-Loop из таблицы плана. Переход нужен и при
  **отпускании**: политика может хотеть позу далеко от той, где оператор оставил
  кисть. План про это молчит.
- *Относительный режим* — это «сцепление» (clutch) из классической телеоперации.
  В коде он сводится к шагу 3: `pose_difference` оператора, `rotate_offset` в систему
  базы, `apply_offset` к якорю робота.
- *Метка для шага оператора* — обратная операция к `apply_residual`: `pose_difference`
  и проекция на синергии, делённые на масштабы. Если оператор поправляет сильнее β,
  метка насыщается. Доля `label_saturated` — показатель для выбора β.
- *Базовую политику вызывают и во время перехвата*: без `a_base` нельзя посчитать метку.

---

## Шаг 11. Реальное железо — `real_backend.py`

**Зачем.** Тот же контракт бэкенда, но с UR10e и настоящей кистью.

**Задание.** `RtdeArm` (поток servoL 500 Гц), `HandServo` (поток `servo_tick` 60 Гц),
`RealBackend`, который их объединяет и добавляет камеры LeRobot.

### Решение: команды

Файл `src/lerobot_robot_ur10_dg5f/lerobot_robot_ur10_dg5f/real_backend.py`:

```bash
# в контейнере
cat > /workspace/src/lerobot_robot_ur10_dg5f/lerobot_robot_ur10_dg5f/real_backend.py <<'EOF'
"""Real cell: UR10e through ur_rtde, DG5F through the existing servo controller."""

import threading
import time

import numpy as np
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot_robot_dg5f import Dg5f, Dg5fConfig
from lerobot_robot_dg5f.constants import JOINT_NAMES

from .backend import RobotState


class RtdeArm:
    """Streams the latest TCP target with servoL at the controller rate (500 Hz on e-Series)."""

    def __init__(self, ip: str, servo_hz: float, lookahead_s: float = 0.1, gain: float = 300.0):
        self.ip = ip
        self.dt = 1.0 / servo_hz
        self.lookahead_s = lookahead_s
        self.gain = gain
        self.control = None
        self.receive = None
        self.fault = None
        self._target = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    @property
    def is_connected(self) -> bool:
        return self.control is not None

    def connect(self) -> None:
        import rtde_control
        import rtde_receive

        self.receive = rtde_receive.RTDEReceiveInterface(self.ip)
        self.control = rtde_control.RTDEControlInterface(self.ip)
        self._target = self.tcp()  # seed once from measurement
        self._start_servo()

    def tcp(self) -> np.ndarray:
        return np.asarray(self.receive.getActualTCPPose(), dtype=np.float64)

    def q(self) -> np.ndarray:
        return np.asarray(self.receive.getActualQ(), dtype=np.float64)

    def protective_stop(self) -> bool:
        return bool(self.receive.isProtectiveStopped() or self.receive.isEmergencyStopped())

    def set_target(self, tcp) -> None:
        with self._lock:
            self._target = np.asarray(tcp, dtype=np.float64).copy()

    def move_home(self, q, speed: float = 0.3, acceleration: float = 0.3) -> None:
        """Blocking moveJ; servoing is stopped first because both use the same script."""
        self._stop_servo()
        self.control.moveJ([float(value) for value in q], speed, acceleration)
        self.set_target(self.tcp())
        self._start_servo()

    def disconnect(self) -> None:
        if self.control is None:
            return
        self._stop_servo()
        self.control.stopScript()
        self.control.disconnect()
        self.receive.disconnect()
        self.control = None
        self.receive = None

    def _start_servo(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="rtde-servo", daemon=True)
        self._thread.start()

    def _stop_servo(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self.control.servoStop()

    def _loop(self) -> None:
        while self._running:
            t_start = self.control.initPeriod()
            with self._lock:
                target = self._target.tolist()
            try:
                # speed and acceleration are ignored by servoL; time, lookahead and gain matter.
                self.control.servoL(target, 0.0, 0.0, self.dt, self.lookahead_s, self.gain)
            except Exception as error:  # surface to the env instead of dying silently
                self.fault = error
                self._running = False
                return
            self.control.waitPeriod(t_start)


def _pass_through_guard(proposed):
    # Same as the ROS bridge with current_guard_enabled=false: the hand has its own protection.
    return np.asarray(proposed, dtype=np.float64).copy()


class HandServo:
    """Owns the DG5F SDK in one thread: servo_tick and telemetry reads never run concurrently."""

    def __init__(self, config: Dg5fConfig, arm_pose_max_age_ms: float = 100.0):
        if config.control_mode != "servo":
            raise ValueError("HandServo requires control_mode='servo'")
        self.config = config
        self.arm_pose_max_age_ms = arm_pose_max_age_ms
        self.robot = None
        self.fault = None
        self._target = None
        self._positions = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    @property
    def is_connected(self) -> bool:
        return self.robot is not None and self.robot.is_connected

    def connect(self) -> None:
        self.robot = Dg5f(self.config)
        self.robot.connect()
        self.robot.prepare_arm()  # passive preflight, sends nothing
        pose = self.robot.begin_arm_blend_from_feedback(max_pose_age_ms=self.arm_pose_max_age_ms)
        self._target = pose.copy()
        self._positions = pose.copy()
        self.fault = None
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="dg5f-servo", daemon=True)
        self._thread.start()

    def set_target(self, hand_deg) -> None:
        with self._lock:
            self._target = np.asarray(hand_deg, dtype=np.float64).copy()

    def positions(self) -> np.ndarray:
        with self._lock:
            return self._positions.copy()

    def disconnect(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self.robot is not None and self.robot.is_connected:
            self.robot.hold_position()
            self.robot.disconnect()

    def _loop(self) -> None:
        period = 1.0 / self.config.servo_rate_hz
        next_tick = time.monotonic()
        while self._running:
            with self._lock:
                target = self._target.copy()
            try:
                self.robot.servo_tick(target, guard=_pass_through_guard)
                observation = self.robot.get_observation()
            except Exception as error:  # recovery is explicit (Dg5f.recover), never automatic
                self.fault = error
                self._running = False
                return
            with self._lock:
                self._positions = np.asarray([observation[f"{joint}.pos"] for joint in JOINT_NAMES])
            next_tick += period
            time.sleep(max(0.0, next_tick - time.monotonic()))


class RealBackend:
    def __init__(self, config):
        self.config = config
        self.arm = RtdeArm(config.ur_ip, config.ur_servo_hz, config.ur_lookahead_s, config.ur_gain)
        self.hand = HandServo(Dg5fConfig(
            id=f"{config.id or 'ur10_dg5f'}_hand",
            backend=config.hand_backend,
            ip=config.hand_ip,
            control_mode="servo",
            servo_rate_hz=config.hand_servo_hz,
            disabled_joint_positions_deg=dict(config.hand_disabled_joints_deg),
            calibration_dir=config.calibration_dir,
        ))
        self.cameras = make_cameras_from_configs(config.cameras)
        self._next_tick = None

    @property
    def is_connected(self) -> bool:
        return self.arm.is_connected and self.hand.is_connected

    def connect(self) -> None:
        self.arm.connect()
        try:
            self.hand.connect()
            for camera in self.cameras.values():
                camera.connect()
        except Exception:
            self.disconnect()
            raise
        self._next_tick = time.monotonic()

    def disconnect(self) -> None:
        for camera in self.cameras.values():
            if camera.is_connected:
                camera.disconnect()
        self.hand.disconnect()
        self.arm.disconnect()

    def read(self) -> RobotState:
        return RobotState(
            tcp=self.arm.tcp(),
            arm_q=self.arm.q(),
            hand_deg=self.hand.positions(),
            protective_stop=(self.arm.protective_stop() or self.arm.fault is not None
                             or self.hand.fault is not None),
        )

    def command(self, tcp_target, hand_deg) -> None:
        self.arm.set_target(tcp_target)
        self.hand.set_target(hand_deg)

    def wait_next_period(self, period_s: float) -> None:
        self._next_tick += period_s
        delay = self._next_tick - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        else:
            self._next_tick = time.monotonic()  # overrun: do not try to catch up

    def go_home(self, arm_q) -> None:
        self.arm.move_home(arm_q)
        self._next_tick = time.monotonic()

    def images(self) -> dict[str, np.ndarray]:
        return {name: camera.async_read() for name, camera in self.cameras.items()}
EOF
```

Тесты подменяют модули `rtde_control` и `rtde_receive` подделками, а кисть ставят на
мок-бэкенд (`hand_backend="mock"`). Так проверяются потоки и порядок вызовов, но не
настоящий робот.

Файл `src/lerobot_robot_ur10_dg5f/test/test_real_backend.py`:

```bash
# в контейнере
cat > /workspace/src/lerobot_robot_ur10_dg5f/test/test_real_backend.py <<'EOF'
"""The real backend against fake ur_rtde modules and the DG5F mock: threading, not physics."""

import sys
import threading
import time
import types

import numpy as np
import pytest
from lerobot_robot_dg5f.constants import JOINT_NAMES

from lerobot_robot_ur10_dg5f import Ur10Dg5fConfig
from lerobot_robot_ur10_dg5f.real_backend import RealBackend


START_TCP = [-0.10, 0.50, 0.45, 2.2, -2.2, 0.0]


class FakeReceive:
    def __init__(self, ip):
        self.tcp = list(START_TCP)
        self.q = [0.0] * 6
        self.protective = False

    def getActualTCPPose(self):
        return list(self.tcp)

    def getActualQ(self):
        return list(self.q)

    def isProtectiveStopped(self):
        return self.protective

    def isEmergencyStopped(self):
        return False

    def disconnect(self):
        pass


class FakeControl:
    def __init__(self, ip):
        self.targets = []
        self.servo_active = False
        self.events = []
        self.lock = threading.Lock()

    def initPeriod(self):
        return time.monotonic()

    def waitPeriod(self, t_start):
        time.sleep(max(0.0, t_start + 0.002 - time.monotonic()))

    def servoL(self, pose, speed, acceleration, dt, lookahead, gain):
        assert 0.03 <= lookahead <= 0.2 and 100 <= gain <= 2000
        with self.lock:
            self.servo_active = True
            self.targets.append(list(pose))
        return True

    def servoStop(self, a=10.0):
        self.servo_active = False
        self.events.append("servoStop")
        return True

    def moveJ(self, q, speed, acceleration, asynchronous=False):
        assert not self.servo_active, "moveJ while servoing"
        self.events.append(("moveJ", list(q)))
        return True

    def stopScript(self):
        self.events.append("stopScript")

    def disconnect(self):
        self.events.append("disconnect")


@pytest.fixture
def backend(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "rtde_receive", types.SimpleNamespace(RTDEReceiveInterface=FakeReceive))
    monkeypatch.setitem(sys.modules, "rtde_control", types.SimpleNamespace(RTDEControlInterface=FakeControl))
    config = Ur10Dg5fConfig(id="test", calibration_dir=tmp_path, backend="real", hand_backend="mock")
    backend = RealBackend(config)
    backend.connect()
    yield backend
    backend.disconnect()


def wait_for(condition, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.005)
    return False


def test_servo_thread_streams_the_latest_target(backend):
    target = np.array(START_TCP) + np.r_[0.01, 0, 0, 0, 0, 0]
    backend.command(target, np.zeros(20))
    control = backend.arm.control
    assert wait_for(lambda: control.targets and np.allclose(control.targets[-1], target))


def test_hand_follows_with_servo_velocity_limit(backend):
    target = np.zeros(20)
    target[JOINT_NAMES.index("rj_dg_2_2")] = 30.0
    start = time.monotonic()
    backend.command(np.array(START_TCP), target)
    index = JOINT_NAMES.index("rj_dg_2_2")
    assert wait_for(lambda: abs(backend.read().hand_deg[index] - 30.0) < 0.5, timeout=2.0)
    elapsed = time.monotonic() - start
    assert elapsed >= 30.0 / 120.0 * 0.9  # not faster than servo_max_velocity_deg_s


def test_go_home_stops_servo_before_movej(backend):
    backend.go_home(np.zeros(6))
    events = backend.arm.control.events
    assert events.index("servoStop") < next(i for i, e in enumerate(events) if e[0] == "moveJ")
    assert backend.arm._thread is not None and backend.arm._thread.is_alive()


def test_protective_stop_is_reported(backend):
    backend.arm.receive.protective = True
    assert backend.read().protective_stop


def test_disconnect_stops_script(backend):
    control = backend.arm.control
    backend.disconnect()
    assert "servoStop" in control.events and "stopScript" in control.events
    assert not backend.hand.is_connected
EOF
```

Тесты шага, а затем весь пакет целиком:

```bash
# в контейнере
cd /workspace/src/lerobot_robot_ur10_dg5f
python3 -m pytest -q test/test_real_backend.py
python3 -m pytest -q test
```

Для работы с настоящим роботом понадобится сам `ur_rtde` (в образе его нет). Поставить
и закрепить версию:

```bash
# на хосте, перед работой с роботом
echo 'ur_rtde==1.6.5' >> docker/python-requirements.txt
bash scripts/stack.sh build    # пересобрать образ
```

**Проверка.** `5 passed` для шага и `54 passed` для всего пакета. Тесты шага: поток servo
доносит новую цель; кисть идёт к ступеньке 30° не быстрее 120°/с
(`servo_max_velocity_deg_s`); `moveJ` вызывается только после `servoStop`; защитная
остановка видна в `read()`; `disconnect` делает `servoStop` и `stopScript`.

**Порядок первого запуска на железе:**

1. **URSim** (Docker-образ `universalrobots/ursim_e-series`): `ur_rtde` работает с ним
   как с роботом. Прогоните на нём шаги 7–10.
2. Реальный UR10e, только чтение (`RTDEReceiveInterface`): сравните
   `getActualTCPPose()` с TCP MuJoCo-бэкенда при тех же суставах. Это проверка
   системы координат из шага 1.
3. Скорость на пульте 10–20 %; там же Safety-плоскости рабочей зоны и предел скорости
   TCP. Это дублирует программные ограничители, и это правильно.
4. Servo с целью «текущая поза» 10 с — робот не должен шевельнуться.
5. Шаги ±2 см по каждой оси. Аварийная кнопка в руке у второго человека.
6. Кисть: сначала `HandServo` с целью «текущая поза» (пальцы не шевелятся), потом
   ступеньки по одному суставу, потом политика.

### Откуда это и где найти самому

- *ur_rtde.* Официальный пример ServoL
  (<https://sdurobotics.gitlab.io/ur_rtde/pages/examples/high_frequency_servoing/servol_example.html>):
  цикл `t_start = rtde_c.initPeriod()` → `servoL(target, velocity, acceleration, dt,
  lookahead_time, gain)` → `rtde_c.waitPeriod(t_start)`, `dt = 1/500`, `lookahead_time =
  0.1`, `gain = 800`, в конце `servoStop()` и `stopScript()`. Сигнатуры и диапазоны —
  `help(...)` после `pip install ur_rtde` (1.6.5): у `servoL` скорость и ускорение
  помечены *«NOT used in current version»*, `moveJ(q, speed=1.05, acceleration=1.4,
  asynchronous=False)` блокирующий. Начинаем с `gain = 300`: мягче, чем в примере, но
  в разрешённом диапазоне.
- *Почему отдельный поток.* Контроллер ждёт команды на своей частоте (500 Гц), а среда
  работает на 10 Гц. Поток шлёт последнюю цель, среда только её обновляет.
- *moveJ и servo в одном скрипте.* Перед `moveJ` servo-поток останавливают и делают
  `servoStop()`: оба используют один скрипт на контроллере.
- *Кисть.* Защиту по току в мосте отключили сознательно: у кисти есть встроенная.
  `Dg5f.send_action` в режиме servo бросает исключение
  (`src/lerobot_robot_dg5f/lerobot_robot_dg5f/servo_controller.py:212`): servo-выход идёт
  только через `servo_tick(target, guard=...)`. Как это делает мост — `_send_servo_tick`
  в `ros_bridge_node.py` (около строки 932). При `current_guard_enabled=false` его guard
  возвращает цель без изменений (`ros_bridge_node.py:514`), и решение делает то же
  самое. ARM-последовательность (`prepare_arm` → `begin_arm_blend_from_feedback`) и
  `arm_pose_max_age_ms: 100.0` взяты оттуда же и из `config/bridge.params.yaml`.
- *Один поток владеет SDK.* В `backends.py` кисти нет блокировок, поэтому `servo_tick` и
  чтение телеметрии выполняет один и тот же поток, а среда берёт кэш под замком.
- *Ошибки.* Защитная остановка UR и сбой SDK кисти заканчивают эпизод
  (`protective_stop` → `terminated`), а восстановление — только явное, как в мосте.

---

## Шаг 12. Встроить в общий прогон и записать выводы

**Зачем.** Чтобы `stack.sh test` проверял новый пакет вместе с остальными.

**Задание.** Добавьте пакет в общий прогон тестов, соберите всё и подготовьте коммит.

### Решение: команды

Добавить пакет в общий прогон тестов (команда ничего не делает, если строка уже есть):

```bash
# в контейнере
cd /workspace
grep -q lerobot_robot_ur10_dg5f scripts/test_inside.sh || \
  sed -i '/^run_package_tests \/workspace\/src\/lerobot_robot_dg5f$/a MUJOCO_GL=egl run_package_tests /workspace/src/lerobot_robot_ur10_dg5f' scripts/test_inside.sh
grep -n run_package_tests scripts/test_inside.sh
```

Полная сборка, весь прогон и подготовка коммита:

```bash
# на хосте, из корня репо
bash scripts/stack.sh compile
bash scripts/stack.sh test
git add src/lerobot_robot_ur10_dg5f models/ur10e_dg5f THIRD_PARTY.md scripts/test_inside.sh docs/UR10_DG5F_ENV_GUIDE.md
git status --short
```

Выводы, которых не видно из кода (выбранные β, пределы шага, найденные ловушки), —
в `JOURNAL.md`.

**Проверка.** `grep` показывает новую строку сразу после `lerobot_robot_dg5f`; в выводе
`stack.sh test` для `lerobot_robot_ur10_dg5f` — `54 passed`; `git status --short`
показывает в индексе только ваши пути, без `isaaclab_ext/`.

### Откуда это и где найти самому

`scripts/test_inside.sh` запускает `run_package_tests` для каждого
пакета после `source /workspace/install/setup.bash`, поэтому сначала нужен
`stack.sh compile`.

---

## Что дальше, за пределами среды

1. **Телеоператор Quest** (`lerobot_teleoperator_quest`): реализует `OperatorSource` и
   контракт `Teleoperator`. Сначала выясните, откуда брать позу запястья: в
   `ManoLandmarks` только 21 точка. Если точки в мировой системе Quest, поза запястья
   считается по точкам 0, 5 и 17; если они уже относительно запястья, позу должно слать
   Unity-приложение.
2. **Запись демонстраций** через `Ur10Dg5f` и телеоператор в LeRobot Dataset (шаг 2 плана).
3. **HIL-SERL.** `lerobot/rl/actor.py` создаёт среду через `make_robot_env` из
   `gym_manipulator`, а та требует `bus.motors`. Понадобится своя фабрика среды для
   actor: небольшой форк `actor.py` или подмена `make_robot_env` и `make_processors`.

## Сводка ловушек

| Ловушка | Где | Как найдена | Что делать |
| --- | --- | --- | --- |
| Мир MuJoCo ≠ база UR (поворот на π) | 1, 6 | `xquat` тела `base`; `ur_macro.xacro` | переводить в бэкенде; сверить с реальным TCP |
| `PYTHONPATH` не делает пакет плагином | 2 | исходник `register_third_party_plugins` | colcon или `pip install -e` |
| rotvec неоднозначен около π | 3, 8 | тест сравнения поз | сравнивать матрицами; в state — 6D |
| Шаг от измеренной позы, а не от команды | 4 | принцип `command_shaper` кисти | считать от последней отправленной |
| Home-поза коллеги на 8 см в столе | 5, 6 | контакты пальцев со столом | home выше на 10 см |
| Нет компенсации гравитации | 6 | `actuator_force` локтя = 150 | `gravcomp` через `MjSpec` до компиляции |
| `body_gravcomp` после компиляции не работает | 6 | дрейф не изменился, `ngravcomp = 0` | только `MjSpec` |
| `scene.xml` перезаписывает интегратор на RK4 | 6 | сила +56 против формулы −184 | `implicitfast` |
| `Dg5f.send_action` не работает в servo | 11 | `servo_controller.py:212` | поток 60 Гц, `servo_tick` со сквозным guard |
| SDK кисти без блокировок | 11 | `backends.py` | один поток и пишет, и читает |
| `moveJ` во время servo | 11 | один скрипт на контроллере | сначала `servoStop` |
| Рывок пальцев при отпускании перехвата | 10 | план молчит | интерполяция и в обратную сторону |
| Метка residual насыщается | 10 | `residual_label` | логировать `label_saturated`, подбирать β |
| Незакоммиченные правки Isaac Lab на ветке | 0 | `git status` | `git add` только своих путей |
