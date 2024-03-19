# wf-pose-engine

Generate and store 2D poses

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

1. First build the container
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


```
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
export GPU_DEVICE_NAME="rtx2080"
python mmdeploy/tools/deploy.py \
    configs/runtimes/mmdeploy/configs/mmpose/pose-detection_rtmo_tensorrt-fp16_dynamic-640x640.py \
    configs/body_2d_keypoint/rtmo/body7/rtmo-l_16xb16-600e_body7-640x640.py \
    checkpoints/rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.pth \
    pose_engine/assets/coco_image_example.jpg \
    --work-dir mmdeploy_model/pose-detection_rtmo_tensorrt-fp16_dynamic-640x640-${GPU_DEVICE_NAME} \
    --device cuda \
    --dump-info
```

| Once the model is converted, the folder `mmdeploy_model/rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211-fp16` can be staged on S3 for deployment