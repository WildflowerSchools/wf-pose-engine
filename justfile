install:
    poetry run pip install chumpy
    poetry install
    poetry run mim install mmcv==2.1.0

format:
    black pose_engine

lint:
    pylint pose_engine

test:
    pytest tests/

version:
    poetry version