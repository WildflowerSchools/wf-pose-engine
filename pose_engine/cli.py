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


def click_options_pose_engine_config():
    return click_add_options(
        [
            click.option(
                "--inference-mode",
                type=click.Choice(["onestage", "topdown"], case_sensitive=False),
                required=False,
                help="Select inference mode/workflow",
                show_envvar=True,
                default="onestage",
                show_default=True,
            ),
            click.option(
                "--detector-batch-size",
                type=int,
                required=False,
                help="Maxiumum numbers of objects the object detector will process per batch",
                show_envvar=True,
            ),
            click.option(
                "--pose-estimator-batch-size",
                type=int,
                required=False,
                help="Maxiumum numbers of objects the pose estimator will process per batch",
                show_envvar=True,
            ),
            click.option(
                "--use-fp16",
                type=bool,
                required=False,
                help="Enable to use float16. Should yield faster results at some risk of impacting accuracy",
                show_envvar=True,
                default=True,
                show_default=True,
            ),
            click.option(
                "--compile",
                type=click.Choice(["inductor", "tensorrt"], case_sensitive=False),
                required=False,
                help="Set to run torch.compile against models before running inference",
                show_envvar=True,
                default=None,
                show_default=True,
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
@click_options_pose_engine_config()
def cli_run(
    environment,
    start,
    end,
    inference_mode,
    detector_batch_size,
    pose_estimator_batch_size,
    use_fp16,
    compile,
):
    from . import core

    core.run(
        environment=environment,
        start=start,
        end=end,
        inference_mode=inference_mode,
        detector_batch_size=detector_batch_size,
        pose_estimator_batch_size=pose_estimator_batch_size,
        use_fp16=use_fp16,
        compile_engine=compile,
    )


@click.command(name="batch", help="Generate poses for a batch of instances")
@click_options_pose_engine_config()
def cli_batch(
    inference_mode, detector_batch_size, pose_estimator_batch_size, use_fp16, compile
):
    from . import core

    core.batch(
        detector_batch_size=detector_batch_size,
        pose_estimator_batch_size=pose_estimator_batch_size,
    )


cli.add_command(cli_run)
cli.add_command(cli_batch)
