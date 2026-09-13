
ARG VARIANT="ubuntu22.04"
ARG CUDA_VERSION="12.4.1"
FROM nvidia/cuda:${CUDA_VERSION}-devel-${VARIANT}

# # set non interactive mode
ARG DEBIAN_FRONTEND=noninteractive

# Let nvidia-container-runtime expose the Vulkan/GL driver bits (nvidia_icd.json,
# libGLX_nvidia, etc.) inside the container. Without "graphics" in here, CARLA's
# -RenderOffScreen mode can't find a Vulkan device and fails to start.
ENV NVIDIA_DRIVER_CAPABILITIES=all

# # install dependencies to build python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcairo2-dev \
    pkg-config \
    python3-dev \
    git \
    curl \
    bzip2 \
    bash-completion \
    ssh \
    sudo \
    vim \
    tmux \
    htop \
    unzip \
    netcat-openbsd \
    psmisc \
    procps \
    libvulkan1 \
    libsdl2-2.0-0 \
    libsm6 \
    libgl1 \
    libomp5 \
    xdg-user-dirs

# nvidia-container-toolkit mounts the real NVIDIA Vulkan driver (libGLX_nvidia.so.0)
# into the container via the "graphics" capability, but on this host it does not
# also mount the tiny ICD manifest that tells the Vulkan loader to use it - so
# without this file, `vkEnumeratePhysicalDevices` finds zero devices and CARLA's
# -RenderOffScreen mode fails to start. The manifest just references the driver
# lib by name; the actual .so always comes from the host driver at container
# start, so this is safe regardless of which driver version is running.
RUN mkdir -p /usr/share/vulkan/icd.d && \
    printf '%s\n' \
    '{' \
    '    "file_format_version" : "1.0.1",' \
    '    "ICD": {' \
    '        "library_path": "libGLX_nvidia.so.0",' \
    '        "api_version" : "1.4.329"' \
    '    }' \
    '}' > /usr/share/vulkan/icd.d/nvidia_icd.json


# add your own dependencies here!
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     [your-own-dependencies] \


### tmux commented out because there is no tmux file and i dont know if i need it
#  COPY config/.tmux.conf /home/${USERNAME}/.tmux.conf


# Non-root user with passwordless sudo. On container start the devcontainer CLI
# remaps this user's UID/GID to the host user's (updateRemoteUserUID) and chowns
# the home directory, which is why uv and conda live under $HOME below.
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=1000
RUN set -eux; \
    if ! getent group "${USER_GID}" > /dev/null; then \
        groupadd --gid "${USER_GID}" "${USERNAME}"; \
    fi; \
    if ! id -u "${USERNAME}" > /dev/null 2>&1; then \
        useradd --uid "${USER_UID}" --gid "${USER_GID}" -m --shell /bin/bash "${USERNAME}"; \
    fi; \
    echo "${USERNAME} ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME}; \
    chmod 0440 /etc/sudoers.d/${USERNAME}

# Install uv system-wide
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN UV_INSTALL_DIR=/usr/local/bin sh /uv-installer.sh && rm /uv-installer.sh

USER ${USERNAME}
WORKDIR /home/${USERNAME}

# Ensure conda is on the `PATH`
ENV CONDA_DIR="/home/${USERNAME}/miniconda"
ENV PATH="$CONDA_DIR/bin:$PATH"

# Install Miniconda and clean up cached files
RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p "$CONDA_DIR" \
    && rm /tmp/miniconda.sh \
    && "$CONDA_DIR"/bin/conda clean -afy \
    && "$CONDA_DIR"/bin/conda init bash

# CarDreamer env, per the main README: python 3.10 + flit (car_dreamer itself is
# installed later via `flit install --symlink` once the repo is mounted at /workspace)
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r \
    && conda create -y -n cardreamer python=3.10 \
    && conda run -n cardreamer pip install --no-cache-dir flit \
    && conda clean -afy \
    && echo "conda activate cardreamer" >> ~/.bashrc


# set DEBIAN_FRONTEND back to dialog
ARG DEBIAN_FRONTEND=dialog
