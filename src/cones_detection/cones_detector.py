import ultralytics
from cones_detection import utils
from config import DETECTOR


class ConesDetector:
    def __init__(self):
        self.detector = ultralytics.YOLO(DETECTOR.joinpath("weights.pt"), "v8")

    def detect(self, image, conf=0.7, save=False):
        detections = self.detector.predict(source=[image], conf=conf, save=save)
        return utils.cuda_tensor_to_cpu(detections[0]).numpy()
