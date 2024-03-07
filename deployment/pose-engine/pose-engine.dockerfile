ARG PYTHON_VERSION=3.11.8
ARG PYENV_ROOT="/etc/pyenv"
ARG POETRY_HOME="/etc/poetry"

FROM nvidia/cuda:12.3.2-devel-ubuntu22.04 as build

ARG PYTHON_VERSION
ARG PYENV_ROOT
ENV PYENV_ROOT=${PYENV_ROOT}
ARG POETRY_HOME
ENV POETRY_HOME=${POETRY_HOME}

ENV DEBIAN_FRONTEND=noninteractive
# LC_ALL=POSIX prevents an issue with CUDA changing the locale during python script runtime
ENV LC_ALL=POSIX

RUN apt update -y && \
    apt install -y software-properties-common \
    build-essential \
    curl \
    ffmpeg \
    git \
    libsm6 \
    libxext6 \
    zlib1g-dev \
    libbz2-dev \
    libffi-dev \
    libssl-dev \
    libreadline-dev \
    libsqlite3-dev \
    liblzma-dev

RUN apt install -y tensorrt

# Install PYENV and set Pyenv's Python version
ENV PATH="${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:$POETRY_HOME/bin:${PATH}"
RUN git clone https://github.com/pyenv/pyenv.git ${PYENV_ROOT} && \
    pyenv install ${PYTHON_VERSION}

# Install NPM (not required to run pose-engine)
# RUN curl -fsSL https://deb.nodesource.com/setup_21.x | bash - && \
#     apt install -y nodejs npm

# Install POETRY
# ENV POETRY_HOME="/etc/poetry" \
    
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
# ENV PATH="${PATH}:${POETRY_HOME}/bin"

RUN curl -sSL https://install.python-poetry.org | python3 -

WORKDIR /app

COPY pyproject.toml poetry.lock ./
RUN eval "$(pyenv init -)" && pyenv local ${PYTHON_VERSION} && \
    poetry config virtualenvs.in-project true --local && \
    poetry env use $(which python) && \
    . $(poetry env info --path)/bin/activate && \
    pip install setuptools wheel six auditwheel && \
    pip install chumpy faster_fifo
RUN poetry install --only main --no-root --no-interaction --no-ansi && \
    poetry run mim install mmcv==2.1.0 && \
    poetry run pip install torch_tensorrt

# FROM nvidia/cuda:12.3.2-devel-ubuntu22.04 as runtime

# ARG PYENV_ROOT
# ENV PYENV_ROOT=${PYENV_ROOT}
# ARG POETRY_HOME
# ENV POETRY_HOME=${POETRY_HOME}

# COPY --from=build ${PYENV_ROOT} ${PYENV_ROOT}
# COPY --from=build ${POETRY_HOME} ${POETRY_HOME}
# COPY --from=build /app /app

# ENV PATH="${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:$POETRY_HOME/bin:${PATH}"

# WORKDIR /app

COPY configs ./configs
COPY pose_engine ./pose_engine

ENTRYPOINT ["poetry", "run", "python", "-m", "pose_engine"]