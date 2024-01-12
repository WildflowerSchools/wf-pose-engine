from datetime import datetime, timezone
import itertools
import os

from dotenv import load_dotenv
import click
import click_log

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
    help="env file path",
)
@click.option("--profile", is_flag=True)
def cli(env_file, profile):
    if env_file is None:
        env_file = os.path.join(os.getcwd(), ".env")

    if os.path.exists(env_file):
        logger.info("Loading env file")
        load_dotenv(dotenv_path=env_file)

    if profile:
        import cProfile
        import pstats
        import atexit

        import yappi

        use_system_profiler = False

        logger.warning("PROFILING MODE ONE")
        pr = None
        pr_output = None
        yappi_output = None
        if use_system_profiler:
            pr = cProfile.Profile()
            pr.enable()
            pr_output = open("/tmp/pose_engine_cprofile.prof", "w")
        else:
            yappi.set_clock_type(
                "wall"
            )  # Use set_clock_type("cput") for cpu time, use set_clock_type("wall") for wall time
            yappi.start()
            yappi_output = open("/tmp/pose_engine_yappi_profile.txt", "w")

        def exit():
            if pr is not None:
                pr.disable()
                pstats.Stats(pr, stream=pr_output).sort_stats(
                    "cumulative"
                ).print_stats()
                pr_output.flush()
                pr_output.close()
            else:
                yappi.stop()
                threads = yappi.get_thread_stats()
                threads.print_all(out=yappi_output)
                for thread in threads:
                    yappi_output.write(
                        "\nFunction stats for (%s) (%d)" % (thread.name, thread.id)
                    )
                    yappi.get_func_stats(ctx_id=thread.id).print_all(out=yappi_output)
                yappi_output.flush()
                yappi_output.close()

            logger.warning("Profiling completed")

        atexit.register(exit)


@click.command(name="run", help="Generate and store poses from classroom video")
@click_options_environment_start_end()
def cli_run(environment, start, end):
    from . import core

    core.run(environment=environment, start=start, end=end)


@click.command(name="batch", help="Generate poses for a batch of instances")
def cli_batch():
    from . import core

    core.batch()


cli.add_command(cli_run)
cli.add_command(cli_batch)
