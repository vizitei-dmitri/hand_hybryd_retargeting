# Обзор проекта DG5F

[Вернуться к краткому README](../README.md) · [Запуск](https://github.com/vizitei-dmitri/hand_hybryd_retargeting/blob/feature/kinect-mano-hybrid-tracker/docs/RUN_GUIDE.md)

## Overview

The central question is how manipulation skill can move between a human hand, a
simulated robotic hand and a physical one. A person can show a behavior quickly
and intuitively, but their hand has a different shape from the robot's. A
simulator can generate large amounts of interaction, but only if it models the
real hand well enough. A learned policy can find strategies that are hard to
write down, but it benefits from structure and good starting points.

The project connects these pieces in a loop:

```text
      human hand motion and demonstrations
                     |
                     v
   geometric, contact-aware retargeting  <------>  teleoperation
                     |
                     v
       robot-space trajectories and datasets
                     |
                     v
   imitation learning  <------>  reinforcement learning
                     |
                     v
        dexterous manipulation policies
                     |
                     v
   simulation  <------ system identification ------>  physical hand
```

The arrows are not a fixed pipeline. Teleoperation is used both to check
retargeting and to produce data. Simulation shapes how policies are trained, and
experiments on the physical hand in turn refine the simulation. Human
demonstrations can inform learning, and learned behavior can feed back into how
human motion is mapped to the robot.

## Dexterous Manipulation and Learning

The target problem is general dexterous manipulation with a multi-finger hand,
not a single benchmark. Relevant behaviors include in-hand rotation and
reorientation of objects, keeping a grasp stable while fingers move, handling
sustained and changing contacts, coordinating many fingers at once, and
adapting to objects of different shapes. A cube held and turned in the palm
is a convenient first task because it exercises most of these at once, but it
is a testbed rather than the goal.

Reinforcement learning and imitation learning are treated as complementary
tools. Reinforcement learning lets manipulation strategies emerge from
interaction and task rewards, which suits contact-rich behavior that is hard to
specify by hand. Imitation learning brings in structure from human
demonstrations or other expert trajectories, which can make exploration more
efficient and keep behavior closer to how a person would perform the task. The
project does not commit to one way of combining them. The same robot interface
and data format are used for recorded demonstrations and policy execution, so
different combinations can be compared.

Combining learning with geometric retargeting is especially interesting.
Optimization gives explicit structure: kinematic feasibility, joint limits,
preserved contacts and meaningful correspondences with the human hand. Learning
can capture what is difficult to write as an objective, such as how much force
to apply, when to regrasp, or how to recover when an object slips.

## Human Hand Retargeting and Teleoperation

Human and robot hands differ in proportions, kinematics, joint layout,
reachable workspace and contact geometry. Copying joint angles from one to the
other therefore gives poor results, and retargeting is posed as an optimization
problem instead: find a robot configuration that preserves the useful parts of
the human motion while respecting the robot's own constraints.

What counts as useful depends on the task. Typical terms include fingertip
positions and their relative geometry, finger directions, the overall hand pose,
the shape of a grasp, spatial relationships between fingers, and contacts with
the object. Pinches and closing grasps matter more than the exact angle of every
joint. The retargeting modules in this repository combine vector-based and
fingertip-distance-based objectives, and add contact-aware corrections when
fingers approach each other.

A broader direction is to move beyond sparse joint landmarks towards richer
geometric representations. Correspondences between human and robot hands can be
defined on surfaces and meshes, on local geometric features, on contact regions,
or through task-space constraints tied to the manipulated object. The goal is
not to reproduce the human hand literally, but to preserve the geometry and
intent of the manipulation even though the robot's morphology is different.

Teleoperation is the most direct use of retargeting. Hand tracking from a VR
headset provides human hand observations, retargeting turns them into joint
commands, and the robot follows in simulation or on the physical hand. Beyond
manual control, this setup is used to collect demonstrations, observe how
people solve manipulation tasks, evaluate retargeting quality, generate
trajectories for learning, and eventually combine human guidance with
autonomous control.

## Simulation and Real-World Transfer

Dexterous manipulation involves many simultaneous contacts and a
high-dimensional action space, so learning directly on hardware is slow and
risky. Simulation makes it possible to collect large amounts of interaction in
parallel and to try control strategies safely.

Simulation is only useful if it reflects the properties of the real hand that
matter for control. These include actuator dynamics, latency, joint limits,
torque and velocity constraints, contact behavior and friction. System
identification from recordings of the physical hand is used to fit these models,
and the same control interface is kept across simulation and hardware. This
makes sim-to-real transfer a modeling problem that can be studied directly,
rather than a final deployment step.

## Repository

```text
hand_hybryd_retargeting/
├── compose.yaml, docker/        # containerized ROS 2 + LeRobot environment
├── scripts/stack.sh             # build, launch, test and data-collection entry point
├── src/
│   ├── dg5f_teleop/             # hand retargeting and MuJoCo visualization
│   ├── dg5f_unity_teleop/       # VR hand-tracking adapter and combined launch
│   ├── lerobot_robot_dg5f/      # LeRobot robot plugin, command shaping, dataset recording
│   ├── ros_tcp_endpoint/        # ROS TCP bridge for the VR application
│   └── vr_haptic_msgs/
├── isaaclab_ext/dg5f_isaaclab/  # Isaac Lab environments for learning in simulation
├── models/dg5f/                 # hand model and meshes
├── vendor/tesollo_control/      # hand SDK and Python bindings
└── docs/                        # additional guides
```
