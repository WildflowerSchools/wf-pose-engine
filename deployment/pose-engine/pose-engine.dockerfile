FROM nvidia/cuda:12.3.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive 

RUN apt update -y && \
    apt install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt install -y \
    build-essential \
    curl \
    # python
    python3.11-dev \
    python3-pip \
    # cv2 dependencies
    ffmpeg \ 
    libsm6 \
    libxext6

# INSTALL NPM (not required to run pose-engine)
# RUN curl -fsSL https://deb.nodesource.com/setup_21.x | bash - && \
#     apt install -y nodejs npm

# INSTALL POETRY
ENV POETRY_HOME=/etc/poetry \
    PATH="${PATH}:/etc/poetry/bin" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

RUN curl -sSL https://install.python-poetry.org | python3 -

WORKDIR /build

COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false && \
    poetry run pip install chumpy faster_fifo && \
    poetry install --only main --no-root --no-interaction --no-ansi && \
    poetry run mim install mmcv==2.1.0

WORKDIR /app

COPY configs ./configs
COPY pose_engine ./pose_engine

ENTRYPOINT ["python3", "-m", "pose_engine"]