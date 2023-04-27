import argparse
import logging
import os
import pathlib
import sys
import typing
import cv2

from camera_perception import logs, common
from camera_perception.camera import Camera, CameraError
from cones_detection.cones_detector import ConesDetector
from cones_detection.landmark import ConeLandmark


def get_help_epilog():
    return """
Exit codes:
    0 - successful execution
    1 - missing input file
    2 - not implemented option
    3 - unable to open camera device
    any other code indicated unrecoverable error

Environment variables:
    CONFIDENCE : Model detection confidence threshold
                 Default: 0.75

More info: <https://github.com/lukaszmichalskii/camera-perception>"""


def run_app(
    args: argparse.Namespace,
    argv: typing.List[str],
    logger: logging.Logger,
    environment: common.Environment,
) -> int:
    def bounding_box():
        cv2.rectangle(
            frame,
            cone.bounding_box.top_left,
            cone.bounding_box.bottom_right,
            cone.color,
            3
        )

    def label():
        cv2.putText(
            frame,
            f"{cone.name} {round(cone.conf * 100)}%",
            (cone.bounding_box.top_left[0], cone.bounding_box.top_left[1] - 10),
            font,
            0.5,
            cone.color,
            1
        )

    if environment.os != "linux":
        logger.warning(
            f"You are using toolkit on {environment.os}. Some functionalities may not work correctly"
        )
    logger.info(
        f"cones_detector: {sys.executable} argv: {argv} {environment.to_info_string()}"
    )

    video = pathlib.Path(args.video) if args.video else None
    if video and not video.exists():
        logger.error(f"Video file '{str(video)}' not exist.")
        logger.info("App finished with exit code 1")
        return 1

    if args.image:
        logger.info(f"Not implemented.")
        return 2

    font = cv2.FONT_HERSHEY_DUPLEX
    camera_lens = Camera(args.video)
    cones_detector = ConesDetector()
    try:
        for frame in camera_lens.get_frame():
            detections = cones_detector.detect(image=frame, conf=environment.conf)
            if len(detections) > 0:
                for detection in detections.boxes:
                    cone = ConeLandmark(
                        cls=int(detection.cls[0]),
                        conf=detection.conf[0],
                        detection=detection
                    )
                    bounding_box()
                    label()

            cv2.imshow("Cones Detection", frame)
            if cv2.waitKey(1) == ord("q"):
                break

        camera_lens.turn_off()
        cv2.destroyAllWindows()
    except CameraError as e:
        logger.error(str(e))
        return 3

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
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--video",
        type=str,
        metavar="input_data",
        help="specifies video with environment for cones detection",
    )
    group.add_argument(
        "--image",
        type=str,
        metavar="input_data",
        help="specifies image with environment for cones detection",
    )
    parser.epilog = get_help_epilog()
    return run_app(parser.parse_args(argv[1:]), argv, logger, environment)
