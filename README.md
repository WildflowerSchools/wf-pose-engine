# wf-pose-engine

Generate and store 2D poses

## Install

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