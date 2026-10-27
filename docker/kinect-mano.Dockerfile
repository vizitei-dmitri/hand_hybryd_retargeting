# syntax=docker/dockerfile:1
ARG ROS_BASE_IMAGE=dg5f-lerobot-retargeting:humble

# Independent modern CPU environment. No ROS, LeRobot or legacy CUDA wheels.
FROM python:3.10-slim AS core
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir --upgrade "pip>=25,<27" \
    && python -m pip install --no-cache-dir "setuptools==79.0.1" wheel \
    && python -m pip install --no-cache-dir "torch==2.10.0+cpu" \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
    && python -m pip install --no-cache-dir \
        "numpy==2.2.6" "scipy==1.15.3" "opencv-contrib-python==4.12.0.88" \
        "matplotlib==3.10.9"
COPY kinect_mano/pyproject.toml /opt/kinect_mano/pyproject.toml
COPY kinect_mano/kinect_mano /opt/kinect_mano/kinect_mano
COPY kinect_mano/config /opt/kinect_mano/config
RUN python -m pip install --no-cache-dir --no-deps --no-build-isolation -e /opt/kinect_mano
WORKDIR /data
ENTRYPOINT ["python", "-m", "kinect_mano"]
CMD ["--help"]

# Optional standalone test image; production core has no pytest dependency.
FROM core AS test
RUN python -m pip install --no-cache-dir "pytest>=8,<9"
COPY kinect_mano/tests /opt/kinect_mano/tests
WORKDIR /opt/kinect_mano
ENTRYPOINT ["python", "-m", "pytest"]
CMD ["-q", "tests"]

# Overlay the known local ROS image. Its Torch, CUDA and numerical stack remain
# in the parent image; only the new project and visualization extras enter a venv.
FROM ${ROS_BASE_IMAGE} AS ros
USER root
SHELL ["/bin/bash", "-c"]
RUN apt-get update && apt-get install -y --no-install-recommends python3-venv \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv --system-site-packages /opt/kinect-mano-ros
ENV PATH="/opt/kinect-mano-ros/bin:${PATH}"
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
COPY kinect_mano/pyproject.toml /workspace/kinect_mano/pyproject.toml
COPY kinect_mano/kinect_mano /workspace/kinect_mano/kinect_mano
COPY kinect_mano/config /workspace/kinect_mano/config
RUN python3 -c 'import json; from importlib.metadata import version; json.dump({n:version(n) for n in ("torch","numpy","scipy")},open("/tmp/kinect-base-versions.json","w"))' \
    && python3 -m pip install --no-cache-dir --no-deps \
        "matplotlib==3.10.9" "opencv-contrib-python==4.12.0.88" \
    && python3 -m pip install --no-cache-dir --no-deps --no-build-isolation \
        -e /workspace/kinect_mano \
    && python3 -c 'import json; from importlib.metadata import version; before=json.load(open("/tmp/kinect-base-versions.json")); assert before == {n:version(n) for n in before}, "The parent numerical stack changed"' \
    && rm /tmp/kinect-base-versions.json
COPY src /workspace/src
COPY models /workspace/models
RUN source /opt/ros/humble/setup.bash \
    && colcon build --base-paths /workspace/src --symlink-install \
        --build-base /workspace/build --install-base /workspace/install \
        --event-handlers console_direct+
USER hand
WORKDIR /workspace
# This is the existing base-image entrypoint; it sources ROS and /workspace/install.
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["ros2", "launch", "kinect_mano_ros", "kinect_mano.launch.py", "reconstruction:=/data/reconstruction.npz"]
