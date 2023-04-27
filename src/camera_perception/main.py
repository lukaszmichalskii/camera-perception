import argparse
import logging
import os
import pathlib
import sys
import typing
import cv2

from camera_perception import logs, common
from camera_perception.camera import Camera


def get_help_epilog():
    return """
Exit codes:
    0 - successful execution
    1 - unable to open camera stream
    any other code indicated unrecoverable error - detections history might be invalid
    
More info: <https://github.com/lukaszmichalskii/camera-perception>"""


def run_app(
    args: argparse.Namespace,
    argv: typing.List[str],
    logger: logging.Logger,
    environment: common.Environment,
) -> int:
    if environment.os != "linux":
        logger.warning(
            f"You are using toolkit on {environment.os}. Some functionalities may not work correctly"
        )
    logger.info(
        f"pythonApp: {sys.executable} argv: {argv} {environment.to_info_string()}"
    )
    if os.path.exists(args.output) and os.listdir(args.output):
        logger.error(f"Output directory {args.output} is not empty.")
        logger.info("App finished with exit code 1")
        return 1

    output = pathlib.Path(args.output)
    if output and not output.exists():
        logger.info(
            f"Path '{str(output.absolute())}' not exist, creating results storage space {output.absolute()}..."
        )
        output.mkdir()

    video = pathlib.Path(args.video)
    if not video.exists():
        logger.error(f"Video file '{str(video)}' not exist.")
        logger.info("App finished with exit code 1")
        return 1

    camera_lens = Camera(args.video)
    for frame in camera_lens.get_frame():
        cv2.imshow("Cones Detection", frame)
        if cv2.waitKey(1) == ord("q"):
            break

    camera_lens.turn_off()
    cv2.destroyAllWindows()

    logger.info("App finished with exit code 0")
    return 0


def main(argv: typing.List[str], logger=None, environment=None) -> int:
    if logger is None:
        logger = logs.setup_logger()
    if environment is None:
        environment = common.Environment.from_env(os.environ)
    parser = argparse.ArgumentParser(
        description="Camera Perception - deep learning based computer vision system for cones detection.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        metavar="input_data",
        help="specifies video with environment for cones detection",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        metavar="output_folder",
        default="results",
        help="specifies directory, where results should be saved. Has to be empty",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Display extended information about system execution.",
    )
    parser.epilog = get_help_epilog()
    return run_app(parser.parse_args(argv[1:]), argv, logger, environment)
