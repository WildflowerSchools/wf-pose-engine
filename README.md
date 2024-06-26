# wf-pose-engine

Generate and store 2D poses

## Performance

> Note, performance is degraded because we prepare video -> frames during execution of pose_engine. If we bypassed that step and took strain off the CPU/GPU, we'd see ~10-20% improved FPS.

> Also note, the following uses the POSE_ENGINE_VIDEO_USE_CUDACODEC_WITH_CUDA=true option to move video->frame processing to the GPU

* AWS g5.2xlarge (A10G)
    * Dense: 128.36 FPS
    * Sparse: 152.09 FPS
* AWS g5.4xlarge (A10G)
    * Dense: 144.21 FPS
    * Sparse: 173.37 FPS
* AWS p3.2xlarge (V100)
    * Dense: 94.51 FPS (CPU bound due to video -> frame processing)
    * Sparse: 98.73 FPS

_"sparse frames" =  ~1 person per frame_
_"dense frames" =  ~16 people per frame_

## Run Pose Generation

```
poetry run python -m pose_engine \
       --verbosity INFO \
       --env-file .env \
       run \
       --environment dahlia \
       --start 2023-11-10T14:10:33-0700 \
       --end 2023-11-10T14:10:35-0700
```

## Run Pose Generation using Docker

1. Build the container

    a. Before building, you need to install two packages on your host machine. `libnvidia-decode-XXX and libnvidia-encode-XXX`. Be sure they match your Nvidia driver version.

    > e.g. `apt install libnvidia-decode-545 libnvidia-encode-545`

    b. You then need to download the [Nvidia Video Codec interface headers](https://developer.nvidia.com/nvidia-video-codec-sdk/download) (this must be done manually from Nvidia's website)

    ```
    bsdtar --strip-components=1 -xvf Video_Codec_SDK_12.2.72.zip -C ./nvidia-video-codec
    ```

    c. Run build

    ```
    just build-docker
    ```

2. Then create a .docker.env file in the project root and fill in the req'd vars
```
cp .env.template .docker.env
```

3. Then run the pose engine CLI service
> Included is a volume mapping that exposes the local wf_pose_engine cache directory and the local mmLab's torch cache directory to the docker application's cache directory
```
docker compose -f stack.yml --profile cli-only \
       run \
       -v /home/${USER}/.cache/wf_pose_engine:/root/.cache/wf_pose_engine \
       -v /home/${USER}/.cache/torch:/root/.cache/torch \
       pose-engine \
       --verbosity INFO \
       run \
       --environment dahlia \
       --start 2023-09-20T17:30:00-0000 \
       --end 2023-09-20T17:31:00-0000
```

## Install Locally

1. Setup python environment

> Assumes `pyenv` is installed

```
pyenv local 3.11.5
pyenv virtualenv 3.11.5 wf-pose-engine
pyenv local wf-pose-engine
```

2. Setup npm environment

> Assumes `nvm` is installed

```
nvm install 21.1.0
nvm use 21.1.0
npm install
```

3. Install dependencies

> Assumes `just` is installed

```
just install
```

4. Create Mongo service

> Assumes `docker` is installed

```
just run-docker
```

### Run TensorRT Model

> Download tar'd tensorrt (8.6.1.6) from Nvidia and uncompress

```
pip install TensorRT-8.6.1.6/python/tensorrt-8.6.1-cp311-none-linux_x86_64.whl
```

> Download cudnn (8.9.7.29) from Nvidia and uncompress

#### Setup ENV vars

>Update .env with:

```
TENSORRT_DIR=<<PATH>>
CUDNN_DIR=<<PATH>>
LD_LIBRARY_PATH=$CUDNN_DIR/lib:$LD_LIBRARY_PATH
```

## Test

> Test video was generated from ForBiggerFun open source video. Video was cut into 10 second clips at 10 FPS using the following cmds:

```
cd /tmp
mkdir -p output
curl -O http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerFun.mp4
ffmpeg -i ForBiggerFun.mp4 -fps_mode cfr -filter:v fps=fps=10 -c:v libx264 -x264opts fps=10:keyint=10:min-keyint=10:scenecut=0 ForBiggerFun_10fps.mp4
ffmpeg -i ForBiggerFun_10fps.mp4 -c copy -map 0 -segment_time 00:00:10 -f segment -reset_timestamps 1 ./output/output%03d.mp4
```

To verify frame counts = 100, use:

```
ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames output/output000.mp4
```

## Migrations

> We're using `migrate-mongo` to manage the Mongo db and collections

### CREATE MIGRATION

```
just migrate-create <<description>>
```

### RUN MIGRATIONS

> Be sure Mongo is running and `MONGO_POSE_URI` is set in the `.env` file

```
just migrate
```

### INITIAL SETUP

The migrations folder was initially created with:

```
mkdir -p migrate-mongo
cd migrate-mongo
npx migrate-mongo init
npx migrate-mongo create pose_2d_collection
```

The `url` attribute was updated to: **process.env.MONGO_POSE_URI,** and the first migration file was filled in to create the **poses_2d** collection

## Compile model for ONNX or TensorRT

### Fetch mmDeploy for building model
```
apt update
apt install -y git vim wget
git clone -b main https://github.com/open-mmlab/mmdeploy.git
wget https://download.openmmlab.com/mmpose/v1/projects/rtmo/rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.pth -P checkpoints/
```

### Compile RTMO to ONNX + FP16
```
python mmdeploy/tools/deploy.py \
    configs/runtimes/mmdeploy/configs/mmpose/pose-detection_rtmo_onnxruntime_dynamic-fp16.py \
    configs/body_2d_keypoint/rtmo/body7/rtmo-l_16xb16-600e_body7-640x640.py \
    checkpoints/rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.pth \
    pose_engine/assets/coco_image_example.jpg \
    --work-dir mmdeploy_model/rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211-fp16 \
    --device cuda \
    --dump-info
```

### Compile RTMO to TensorRT + FP16
> Note, tensorrt models must be compiled for each GPU version being targeted (note the GPU_DEVICE_NAME env var)
```
export GPU_DEVICE_NAME="v100"
python mmdeploy/tools/deploy.py \
    configs/runtimes/mmdeploy/configs/mmpose/pose-detection_rtmo_tensorrt-fp16_dynamic-640x640_v100.py \
    configs/body_2d_keypoint/rtmo/body7/rtmo-l_16xb16-600e_body7-640x640.py \
    checkpoints/rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.pth \
    pose_engine/assets/coco_image_example.jpg \
    --work-dir mmdeploy_model/pose-detection_rtmo_tensorrt-fp16_dynamic-640x640-${GPU_DEVICE_NAME} \
    --device cuda \
    --dump-info
```

### Compile RTMO to TensorRT + INT8

#### Prework
```
apt install cuda-toolkit-11-8
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
send to data/coco/annotations/

wget http://images.cocodataset.org/zips/val2017.zip
send to data/coco/val2017
wget http://images.cocodataset.org/zips/train2017.zip
send to data/coco/train2017

pip install h5py
export CPATH=$CPATH:/usr/local/cuda-11/include/:/usr/local/cuda/include:/usr/local/cuda-11/include/
pip install pycuda

apt install cuda-toolkit-12-3
export CPATH=/usr/local/cuda-12.3/include
export PATH=/app/.venv/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/cuda-12.3/lib64
export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda-12.3/lib64
export LIBRARY_PATH=/usr/local/cuda-12.3/lib64
```

#### Compile
> As of 3/20/2024, this does NOT work. I believe it's because of a memory size limitation, but not entirely sure.
```
python mmdeploy/tools/deploy.py  \
    configs/runtimes/mmdeploy/configs/mmpose/pose-detection_rtmo_tensorrt-int8_dynamic-640x640.py \
    configs/body_2d_keypoint/rtmo/body7/rtmo-l_16xb16-600e_body7-640x640.py \
    checkpoints/rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.pth  \
    pose_engine/assets/coco_image_example.jpg \
    --work-dir mmdeploy_model/pose-detection_rtmo_tensorrt-int8_dynamic-640x640-${GPU_DEVICE_NAME} \
    --device cuda \
    --dump-info
```

| Once the model is converted, the folder `mmdeploy_model/rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211-fp16` can be staged on S3 for deployment