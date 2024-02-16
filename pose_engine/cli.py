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
                show_envvar=True,
            ),
            click.option(
                "--start",
                type=click.DateTime(formats=valid_date_formats),
                required=True,
                callback=timezone_aware,
                help="Start time in supported datetime format",
                show_envvar=True,
            ),
            click.option(
                "--end",
                type=click.DateTime(formats=valid_date_formats),
                required=True,
                callback=timezone_aware,
                help="End time in supported datetime format",
                show_envvar=True,
            ),
        ]
    )


@click_log.simple_verbosity_option(logger, show_envvar=True)
@click.group(context_settings={"auto_envvar_prefix": "POSE_ENGINE"})
@click.option(
    "--env-file", type=click.Path(exists=True), help="env file path", show_envvar=True
)
@click.option("--profile", is_flag=True, show_envvar=True)
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

        logger.warning("PROFILING MODE ONE")

        use_system_profiler = False
        pr = None
        pr_output = None
        yappi_output = None
        if use_system_profiler:
            pr = cProfile.Profile()
            pr.enable()
            # pylint: disable=R1732
            pr_output = open(
                file="/tmp/pose_engine_cprofile.prof", mode="w", encoding="utf-8"
            )
        else:
            yappi.set_clock_type(
                "wall"
            )  # Use set_clock_type("cput") for cpu time, use set_clock_type("wall") for wall time
            yappi.start()
            # pylint: disable=R1732
            yappi_output = open(
                file="/tmp/pose_engine_yappi_profile.txt", mode="w", encoding="utf-8"
            )

        def close_profiler():
            try:
                if pr is not None:
                    pr.disable()
                    if pr_output is not None:
                        pstats.Stats(pr, stream=pr_output).sort_stats(
                            "cumulative"
                        ).print_stats()

                if yappi_output is not None:
                    yappi.stop()
                    threads = yappi.get_thread_stats()
                    # pylint: disable=E1101
                    threads.print_all(out=yappi_output)
                    for thread in threads:
                        yappi_output.write(
                            f"Function stats for ({thread.name}) ({thread.id})"
                        )
                        # pylint: disable=E1101
                        yappi.get_func_stats(ctx_id=thread.id).print_all(
                            out=yappi_output
                        )
            finally:
                if pr_output is not None:
                    pr_output.flush()
                    pr_output.close()
                if yappi_output is not None:
                    yappi_output.flush()
                    yappi_output.close()

                logger.warning("Profiling completed")

        atexit.register(close_profiler)


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
