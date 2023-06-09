import sys


if __name__ == "__main__":
    if sys.version_info[:2] < (3, 8):
        sys.exit(
            "Python {}.{}.{} is not supported. You should run app with Python 3.8 or later".format(
                *sys.version_info[:3]
            )
        )
    import camera_perception.main

    sys.exit(camera_perception.main.main(sys.argv))
