import cv2


def bounding_box(image, cone):
    cv2.rectangle(
        image,
        cone.bounding_box.top_left,
        cone.bounding_box.bottom_right,
        cone.color,
        3,
    )


def label(image, cone, font):
    cv2.putText(
        image,
        f"{cone.name} {round(cone.conf * 100)}%",
        (cone.bounding_box.top_left[0], cone.bounding_box.top_left[1] - 10),
        font,
        0.5,
        cone.color,
        1,
    )
