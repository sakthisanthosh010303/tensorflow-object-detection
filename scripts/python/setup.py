# Author: Sakthi Santhosh
# Created on: 01/09/2022
#
# Tensorflow Object Detection - Setup
def main() -> int:
    from os import (
        mkdir,
        path
    )

    # XXX: Relative path provided.
    PATHS = {
        "capture": "./workspace/camera",
        "testing": (
            "./workspace/testing",
            "./workspace/testing/images",
            "./workspace/testing/annotations"
        ),
        "training": (
            "./workspace/training",
            "./workspace/training/images",
            "./workspace/training/annotations"
        ),
        "output": "./workspace/output"
    }

    with open("./objects.txt", 'r') as file_handle:
        LABELS = tuple(map(str.strip, file_handle.readlines()))

    # Setup folders.
    if not path.exists("./workspace"):
        mkdir("./workspace")

    if not path.exists(PATHS["capture"]):
        mkdir(PATHS["capture"])

    if not path.exists(PATHS["output"]):
        mkdir(PATHS["output"])

    if not path.exists(PATHS["testing"][0]):
        mkdir(PATHS["testing"][0])
        mkdir(PATHS["testing"][1])
        mkdir(PATHS["testing"][2])

    if not path.exists(PATHS["training"][0]):
        mkdir(PATHS["training"][0])
        mkdir(PATHS["training"][1])
        mkdir(PATHS["training"][2])

    for label in LABELS:
        folder = path.join(PATHS["capture"], label)
        if not path.exists(folder):
            mkdir(folder)
            mkdir(path.join(folder, "images"))
            mkdir(path.join(folder, "annotations"))

    return 0

if __name__ == "__main__":
    exit(main())
