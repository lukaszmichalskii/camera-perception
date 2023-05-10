import argparse
import logging
import os
import pathlib
import sys
import typing
import cv2

from camera_perception import logs, common, utils
from camera_perception.camera import Camera, CameraError
from cones_detection.cones_detector import ConesDetector
from cones_detection.landmark import ConeLandmark


def get_help_epilog():
    return """
Exit codes:
    0 - successful execution
    1 - missing input file
    3 - unable to open camera device
    any other code indicated unrecoverable error

Environment variables:
    CONFIDENCE : Model detection confidence threshold
                 Default: 0.7
    RESOLUTION : Frame or image resolution, for better performance consider smaller resolutions
                 Default: 1280x720
                 
Usage:
    Run system with video:
    python3 detect.py --video <path_to_video>
    Specify detection confidence threshold 0.5:
    export CONFIDENCE=0.5 && python3 detect.py --video <path_to_video>

More info: <https://github.com/lukaszmichalskii/camera-perception>"""


def run_app(
    args: argparse.Namespace,
    argv: typing.List[str],
    logger: logging.Logger,
    environment: common.Environment,
) -> int:
    def gui_exit(window):
        return cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1

    if environment.os != "linux":
        logger.warning(
            f"You are using toolkit on {environment.os}. Some functionalities may not work correctly"
        )
    logger.info(
        f"cones_detector: {sys.executable} argv: {argv} {environment.to_info_string()}"
    )

    if args.save and os.path.exists(args.output) and os.listdir(args.output):
        logger.error(f"Output directory {args.output} is not empty")
        logger.info("App finished with exit code 1")
        return 1

    video = pathlib.Path(args.video) if args.video else None
    if video and not video.exists():
        logger.error(f"Video file '{str(video)}' not exist.")
        logger.info("App finished with exit code 1")
        return 1

    image = pathlib.Path(args.image) if args.image else None
    if image and not image.exists():
        logger.error(f"Image file '{str(image)}' not exist.")
        logger.info("App finished with exit code 1")
        return 1

    if environment.compute_platform == "CUDA":
        logger.info(environment.cuda_to_info_string())
    elif environment.compute_platform == "CPU":
        logger.info(environment.cpu_to_info_string())
    else:
        logger.warning(
            f"Detected unknown PyTorch installation {environment.compute_platform} {environment.version}. "
            f"Some functionalities may not work correctly"
        )

    # setup
    font = cv2.FONT_HERSHEY_DUPLEX
    title = "Cones Detection"
    exit_key = "q"
    cache = None
    if args.save:
        cache = []

    cones_detector = ConesDetector()

    output = pathlib.Path(args.output) if args.output else None
    if args.save and not output.exists():
        output.mkdir()

    if video:
        camera_lens = Camera(args.video, environment.resolution)
        try:
            for idx, frame in enumerate(camera_lens.get_frame()):
                detections = cones_detector.detect(image=frame, conf=environment.conf)
                if len(detections) > 0:
                    for detection in detections.boxes:
                        cone = ConeLandmark(
                            cls=int(detection.cls[0]),
                            conf=detection.conf[0],
                            detection=detection,
                        )
                        utils.bounding_box(frame, cone)
                        utils.label(frame, cone, font)

                cv2.imshow(title, frame)
                if args.save:
                    cache.append(frame)
                if cv2.waitKey(1) == ord(exit_key):
                    break
                if gui_exit(title):
                    break

            camera_lens.turn_off()
            cv2.destroyAllWindows()
        except CameraError as e:
            logger.error(str(e))
            return 3

    if image:
        img_inference = cv2.imread(str(image))
        inferences = cones_detector.detect(image=img_inference, conf=environment.conf)
        if len(inferences) > 0:
            for inference in inferences.boxes:
                cone = ConeLandmark(
                    cls=int(inference.cls[0]),
                    conf=inference.conf[0],
                    detection=inference,
                )
                utils.bounding_box(img_inference, cone)
                utils.label(img_inference, cone, font)

            cv2.imshow(title, img_inference)
            if args.save:
                cv2.imwrite(
                    str(
                        output.joinpath(
                            f"{image.stem}_detection.{environment.img_graphics_format}"
                        )
                    ),
                    img_inference,
                )
            while True:
                if cv2.waitKey(1) == ord(exit_key):
                    break
                if gui_exit(title):
                    break
            cv2.destroyAllWindows()

    if args.save and video:
        logger.info("Saving images...")
        for idx, frame in enumerate(cache):
            cv2.imwrite(
                str(
                    output.joinpath(f"detection{idx}.{environment.img_graphics_format}")
                ),
                frame,
            )

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
    parser.add_argument(
        "--save", action="store_true", help="save results from image cones recognition"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        metavar="output_dir",
        default="results",
        help="if '--save' option then specifies directory, where results should be saved. Has to be empty",
    )
    parser.epilog = get_help_epilog()
    return run_app(parser.parse_args(argv[1:]), argv, logger, environment)
