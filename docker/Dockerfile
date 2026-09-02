FROM osrf/ros:humble-desktop-full

ARG HOST_UID=1000
ARG HOST_GID=1000

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libegl1 \
        libgl1-mesa-dri \
        libglfw3 \
        libxcursor1 \
        libxi6 \
        libxinerama1 \
        libxrandr2 \
        python3-colcon-common-extensions \
        python3-pip \
        ros-humble-rmw-cyclonedds-cpp \
        ros-humble-rosbag2-storage-mcap \
    && rm -rf /var/lib/apt/lists/*

COPY docker/python-requirements.txt /tmp/python-requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/python-requirements.txt \
    && rm -f /tmp/python-requirements.txt

RUN groupadd --gid "${HOST_GID}" hand \
    && useradd --create-home --uid "${HOST_UID}" --gid "${HOST_GID}" \
        --shell /bin/bash hand

COPY --chmod=755 docker/ros_entrypoint.sh /ros_entrypoint.sh

ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ENV PYTHONUNBUFFERED=1
ENV QT_X11_NO_MITSHM=1
ENV MUJOCO_GL=glfw

USER hand
WORKDIR /workspace

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["sleep", "infinity"]
