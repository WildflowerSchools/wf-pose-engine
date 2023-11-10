from datetime import datetime, timezone
import itertools
import os

from dotenv import load_dotenv
import click
import click_log

from . import core
from .log import logger


now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
valid_date_formats = list(
    itertools.chain.from_iterable(
        map(
            lambda d: [f"{d}", f"{d}%z", f"{d} %Z"],
            ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"],
        )
    )
)


def click_add_options(options):
    def _add_options(func):
        for option in reversed(options):
            func = option(func)
        return func

    return _add_options


def timezone_aware(_ctx, _param, value):
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)

    return value


def click_options_environment_start_end():
    return click_add_options(
        [
            click.option(
                "--environment",
                type=str,
                required=True,
                help="Accepts a single environment id UUID or environment name",
            ),
            click.option(
                "--start",
                type=click.DateTime(formats=valid_date_formats),
                required=True,
                callback=timezone_aware,
                help="Start time in supported datetime format",
            ),
            click.option(
                "--end",
                type=click.DateTime(formats=valid_date_formats),
                required=True,
                callback=timezone_aware,
                help="End time in supported datetime format",
            ),
        ]
    )


@click_log.simple_verbosity_option(logger)
@click.group()
@click.option(
    "--env-file",
    type=click.Path(exists=True),
    help="env file to load environment_name variables from",
)
def cli(env_file):
    if env_file is None:
        env_file = os.path.join(os.getcwd(), ".env")

    if os.path.exists(env_file):
        load_dotenv(dotenv_path=env_file)


@click.command(name="run", help="Generate and store poses from classroom video")
@click_options_environment_start_end()
def cli_run(environment, start, end):
    core.run(environment=environment, start=start, end=end)


cli.add_command(cli_run)
