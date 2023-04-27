# detection colors RBG
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)
ORANGE = (255, 69, 0)
LARGE_ORANGE = (255, 140, 0)
UNKNOWN = (128, 128, 128)


def rgb2bgr(rgb_color):
    return rgb_color[::-1]


class ConeLandmark:
    def __init__(self, cls, conf, detection):
        self.cls = cls
        self.name = ConeLandmark.map_cone_cls_to_name(cls)
        self.color = rgb2bgr(ConeLandmark.cone_color(self.name))
        self.conf = conf
        self.bounding_box = BoundingBox.from_detection(detection)

    @staticmethod
    def cone_color(cone_name):
        if cone_name == "orange_cone":
            return ORANGE
        if cone_name == "large_orange_cone":
            return LARGE_ORANGE
        if cone_name == "blue_cone":
            return BLUE
        if cone_name == "yellow_cone":
            return YELLOW
        return UNKNOWN

    @staticmethod
    def map_cone_cls_to_name(cls):
        if cls == 8:
            return "orange_cone"
        if cls == 9:
            return "large_orange_cone"
        if cls == 7:
            return "blue_cone"
        if cls == 2:
            return "yellow_cone"
        return "unknown_cone"


class BoundingBox:
    def __init__(self, top_left, bottom_right):
        self.top_left = top_left
        self.bottom_right = bottom_right

    @staticmethod
    def from_detection(detection):
        bounding_box = detection.xyxy[0, :]
        return BoundingBox(
            top_left=(int(bounding_box[0]), int(bounding_box[1])),
            bottom_right=(int(bounding_box[2]), int(bounding_box[3])),
        )
