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

## Install Locally

1. Setup python environment

> Assumes `pyenv` is install

```
pyenv local 3.11.5
pyenv virtualenv 3.11.5 wf-pose-engine
pyenv local wf-pose-engine
```

2. Install dependencies

> Assumes `just` is installed

```
just install
```

3. Create Mongo service

> Assumes `docker` is installed

```
just run-docker
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

### RUN MIGRATIONS

> Be sure Mongo is running and `MONGO_POSE2D_URI` is set in the `.env` file
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

The `url` attribute was updated to: **process.env.MONGO_POSE2D_URI,** and the first migration file was filled in to create the **poses_2d** collection