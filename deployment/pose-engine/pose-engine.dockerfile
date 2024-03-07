ARG PYTHON_VERSION=3.11.8
ARG PYENV_ROOT="/etc/pyenv"

FROM nvidia/cuda:12.3.1-devel-ubuntu22.04 as build

ARG PYTHON_VERSION
ARG PYENV_ROOT

ENV DEBIAN_FRONTEND=noninteractive
# LC_ALL=POSIX prevents an issue with CUDA changing the locale during python script runtime
ENV LC_ALL=POSIX

RUN apt update -y && \
    apt install -y software-properties-common \
    build-essential \
    curl \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    zlib1g-dev \
    libbz2-dev \
    libffi-dev \
    libssl-dev \
    libreadline-dev \
    libsqlite3-dev \
    liblzma-dev \
    zlib1g-dev

RUN apt install -y tensorrt

# Install PYENV and set Pyenv's Python version
ENV PATH="${PATH}:${PYENV_ROOT}/shims:${PYENV_ROOT}/bin"
RUN git clone https://github.com/pyenv/pyenv.git ${PYENV_ROOT} && \
    pyenv install ${PYTHON_VERSION} && \
    pyenv global ${PYTHON_VERSION}
RUN eval "$(pyenv init -)" && pyenv local ${PYTHON_VERSION} && pyenv exec pip install poetry setuptools wheel

# Install NPM (not required to run pose-engine)
# RUN curl -fsSL https://deb.nodesource.com/setup_21.x | bash - && \
#     apt install -y nodejs npm

# Install POETRY
# ENV POETRY_HOME="/etc/poetry" \
    
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
# ENV PATH="${PATH}:${POETRY_HOME}/bin"

# RUN curl -sSL https://install.python-poetry.org | python3 -

WORKDIR /app

COPY pyproject.toml poetry.lock ./
RUN mkdir -p .venv
RUN eval "$(pyenv init -)" && pyenv local ${PYTHON_VERSION} && poetry config virtualenvs.in-project true --local && \
    # . $(poetry env info --path)/bin/activate && \
    pyenv exec pip install poetry setuptools wheel six auditwheel pip && \
    pyenv exec pip install chumpy faster_fifo && \
    poetry install --only main --no-root --no-interaction --no-ansi

RUN poetry run mim install mmcv==2.1.0
# RUN poetry env use $(which python) && \
#     . $(poetry env info --path)/bin/activate
    
# RUN poetry run pip install wheel && \
#     poetry run pip install chumpy faster_fifo && \
#     poetry install --only main --no-root --no-interaction --no-ansi && \
#     poetry run mim install mmcv==2.1.0

RUN poetry run ${PYENV_ROOT}/bin/pyenv exec pip install torch_tensorrt

FROM nvidia/cuda:12.3.1-runtime-ubuntu22.04 as runtime

ARG PYENV_ROOT

COPY --from=build ${PYENV_ROOT} ${PYENV_ROOT}
COPY --from=build /app /app

ENV PATH="${PATH}:${PYENV_ROOT}/shims:${PYENV_ROOT}/bin"

WORKDIR /app

COPY configs ./configs
COPY pose_engine ./pose_engine

ENTRYPOINT ["poetry", "run", "python", "-m", "pose_engine"]