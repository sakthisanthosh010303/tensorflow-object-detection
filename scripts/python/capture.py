# Author: Sakthi Santhosh
# Created on: 01/09/2022
#
# Tensorflow Object Detection - Camera
def main() -> int:
    from cv2 import (
        VideoCapture,
        imwrite
    )
    from os import path
    from time import sleep

    CAPTURE_COUNT = 21

    with open("./objects.txt", 'r') as file_handle:
        LABELS = tuple(map(str.strip, file_handle.readlines()))

    camera_handle = VideoCapture(0)

    for label in LABELS:
        print("Capturing images for %s."%(label))
        for counter in range(1, CAPTURE_COUNT + 1):
            flag, frame = camera_handle.read()
            if not flag:
                print("Error: Image capture failed.")
                break
            print("Saving file %s_image%d.jpg."%(label, counter))
            imwrite(
                path.join("./workspace/camera", label, "images", label + "_image%d.jpg"%(counter)),
                frame
            )
            sleep(2)
        print()
        sleep(5)

    camera_handle.release()
    return 0

if __name__ == "__main__":
    exit(main())
