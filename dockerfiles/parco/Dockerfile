# Base Image
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Remove any third-party apt sources to avoid issues with expiring keys.
RUN rm -f /etc/apt/sources.list.d/*.list

# Install some basic utilities & python prerequisites
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && apt-get install -y --no-install-recommends\
    vim \
    curl \
    apt-utils \
    ssh \
    tree \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    make \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    llvm \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    python3-openssl && \
    rm -rf /var/lib/apt/lists/*

# Set up time zone
ENV TZ=Asia/Seoul
RUN sudo ln -snf /usr/share/zoneinfo/$TZ /etc/localtime

# Add config for ssh connection
RUN echo "PasswordAuthentication yes" >> /etc/ssh/sshd_config && \
    echo "PermitEmptyPasswords yes" >> /etc/ssh/sshd_config

# Create a non-root user and switch to it & Adding User to the sudoers File
ARG USER_NAME=user
ARG USER_PASSWORD=0000
RUN adduser --disabled-password --gecos '' --shell /bin/bash $USER_NAME && \
    echo "$USER_NAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USER_NAME && \
    echo "$USER_NAME:$USER_PASSWORD" | chpasswd 
USER $USER_NAME

# All users can use /home/user as their home directory
ENV HOME=/home/$USER_NAME
RUN mkdir $HOME/.cache $HOME/.config && \
    chmod -R 777 $HOME

# Create a workspace directory
RUN mkdir $HOME/workspace
WORKDIR $HOME/workspace

# Set up python environment with pyenv
ARG PYTHON_VERSION=3.10.13
RUN curl https://pyenv.run | bash
ENV PYENV_ROOT="$HOME/.pyenv"
ENV PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH"
ENV eval="$(pyenv init -)"
RUN cd $HOME && /bin/bash -c "source .bashrc" && \
    /bin/bash -c "pyenv install -v $PYTHON_VERSION" && \
    /bin/bash -c "pyenv global $PYTHON_VERSION"

RUN mkdir $HOME/workspace/optimap
WORKDIR $HOME/workspace/optimap

# Install Python Packages
RUN pip install --upgrade pip && \
    pip install setuptools wheel packaging jupyter

COPY dockerfiles/parco/entrypoint.sh .
RUN sudo chmod +x ./entrypoint.sh

COPY .env .
RUN mkdir ${HOME}/artifect

ENTRYPOINT ["/bin/bash","-c","./entrypoint.sh"]
