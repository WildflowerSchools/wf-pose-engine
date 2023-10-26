import os

import click
import click_log
from dotenv import load_dotenv

from . import core
from .log import logger


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


@click.command(name="run", help="Stubbed help description")
def cli_run():
    core.run()


cli.add_command(cli_run)
