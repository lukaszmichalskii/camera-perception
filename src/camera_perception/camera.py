import cv2


class CameraError(Exception):
    pass


class Camera:
    def __init__(self, video, resolution):
        self.frame_width, self.frame_height = resolution
        self.lens = cv2.VideoCapture(video)

    def get_frame(self):
        if not self.lens.isOpened():
            raise CameraError("Camera lens could not be opened")

        while True:
            ret, frame = self.lens.read()
            if not ret:
                break
            frame = cv2.resize(frame, (self.frame_width, self.frame_height))
            yield frame

    def turn_off(self):
        self.lens.release()
