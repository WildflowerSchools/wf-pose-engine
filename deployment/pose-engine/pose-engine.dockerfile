ARG OPENCV_VERSION=80
ARG PYTHON_VERSION=3.11.8

ARG PYENV_ROOT="/etc/pyenv"
ARG POETRY_HOME="/etc/poetry"

FROM nvidia/cuda:12.3.2-devel-ubuntu22.04 as build

ARG MAKEFLAGS=-j8

ARG OPENCV_PYTHON_ROOT="/build/opencv-python"

ARG OPENCV_VERSION
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
    cmake \
    pkg-config \
    curl \
    ffmpeg \
    git \
    libsm6 \
    libxext6 \
    zlib1g-dev \
    libavformat-dev \
    libavcodec-dev \
    libswscale-dev \
    libbz2-dev \
    libffi-dev \
    libssl-dev \
    libreadline-dev \
    libsqlite3-dev \
    liblzma-dev \
    libgtk2.0-dev \
    pkg-config \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libtbb2 \
    libtbb-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libjpeg-turbo8-dev \
    libjpeg8-dev \
    libmlpack-dev \
    libnetcdf-dev \
    libopenblas-dev

RUN apt install -y tensorrt libcudnn8-dev libcudnn8

# Install PYENV and set Pyenv's Python version
ENV PATH="${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:$POETRY_HOME/bin:${PATH}"
RUN git clone https://github.com/pyenv/pyenv.git ${PYENV_ROOT} && \
    pyenv install ${PYTHON_VERSION}

# Install NPM (not required to run pose-engine)
# RUN curl -fsSL https://deb.nodesource.com/setup_21.x | bash - && \
#     apt install -y nodejs npm

# Mount nvidia video codec library
COPY nvidia-video-codec /usr/local/nvidia-video-codec
RUN cp /usr/local/nvidia-video-codec/Interface/*.h /usr/local/cuda/include  # /usr/include
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/nvidia-video-codec/Lib/linux/stubs/x86_64:/usr/local/nvidia/lib:${LD_LIBRARY_PATH}"

# Build CUDA supported OpenCV
WORKDIR ${OPENCV_PYTHON_ROOT}
# RUN echo "/usr/local/cuda/targets/x86_64-linux/lib/stubs" >> /etc/ld.so.conf.d/999_cuda-${CUDA_VERSION}.conf && ldconfig
RUN git clone --branch ${OPENCV_VERSION} --recursive https://github.com/opencv/opencv-python.git ${OPENCV_PYTHON_ROOT}

RUN eval "$(pyenv init -)" && pyenv local ${PYTHON_VERSION} && \
    pip install numpy && \
    export CMAKE_ARGS="-DWITH_FFMPEG=1 -DWITH_CUDA=ON -DENABLE_FAST_MATH=1 -DCUDA_FAST_MATH=1 -DWITH_CUBLAS=1 -DWITH_NVCUVENC=OFF -DCMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs" && \
    export CMAKE_ARGS="${CMAKE_ARGS} -DWITH_NVCUVID=ON -DCUDA_nvcuvid_LIBRARY=/usr/local/nvidia-video-codec/Lib/linux/stubs/x86_64/libnvcuvid.so" && \
    export CMAKE_ARGS="${CMAKE_ARGS} -DBUILD_LIST=\"core cudaarithm calib3d cudacodec cudafeatures2d cudafilters cudaimgproc cudalegacy cudastereo cudawarping cudev dnn dnn_objdetect dnn_superres features2d flann gapi highgui img_hash imgcodecs imgproc ml objdetect photo python3 stitching text video videoio videostab ximgproc xobjdetect xphoto\"" && \
    export CMAKE_ARGS="${CMAKE_ARGS} -DPYTHON3_NUMPY_INCLUDE_DIRS=$(python3 -c 'import numpy; print(numpy.get_include())')" && \
    ENABLE_CONTRIB=1 MAKEFLAGS=${MAKEFLAGS} pip wheel . --verbose

# /usr/local/cuda/lib64:/usr/local/nvidia-video-codec/Lib/linux/stubs/x86_64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

RUN curl -sSL https://install.python-poetry.org | python3 -

WORKDIR /app

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

COPY pyproject.toml poetry.lock ./

RUN eval "$(pyenv init -)" && pyenv local ${PYTHON_VERSION} && \
    poetry env use $(which python)  && \
    poetry run pip install setuptools wheel six auditwheel && \
    poetry run pip install chumpy faster_fifo && \
    poetry install --only main --no-root --no-interaction --no-ansi && \
    poetry run mim install mmcv==2.1.0 && \
    poetry run pip install torch_tensorrt && \
    poetry run pip uninstall -y opencv-python && \
    poetry run pip install ${OPENCV_PYTHON_ROOT}/*.whl && \
    rm -rf $POETRY_CACHE_DIR

FROM nvidia/cuda:12.3.2-runtime-ubuntu22.04 as runtime

ARG PYENV_ROOT
ENV PYENV_ROOT=${PYENV_ROOT}
ARG POETRY_HOME
ENV POETRY_HOME=${POETRY_HOME}

ENV DEBIAN_FRONTEND=noninteractive
# LC_ALL=POSIX prevents an issue with CUDA changing the locale during python script runtime
ENV LC_ALL=POSIX

RUN apt update -y && \
    apt install -y ffmpeg \
    tensorrt \
    libgtk2.0-dev \
    libcudnn8-dev 

ENV PATH=${PATH}:/usr/local/cuda-12.3/lib64 \
    LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-12.3/lib64:/app/.venv/lib/python3.11/site-packages/nvidia/cublas/lib:/app/.venv/lib/python3.11/site-packages/nvidia/cuda_runtime/lib/

COPY --from=build ${PYENV_ROOT} ${PYENV_ROOT}
COPY --from=build /app /app

WORKDIR /app

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

# Install cudablas for cuda 11 to satisfy an mmDeploy + TensorRT dependency
RUN pip install nvidia-cublas-cu11 nvidia-cuda-runtime-cu11 && \
    ln -s libcublasLt.so.11 /app/.venv/lib/python3.11/site-packages/nvidia/cublas/lib/libcublasLt.so.11.0

COPY configs ./configs
COPY pose_engine ./pose_engine

ENTRYPOINT ["python", "-m", "pose_engine"]