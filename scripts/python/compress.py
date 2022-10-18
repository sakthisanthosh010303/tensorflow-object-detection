# Author: Sakthi Santhosh
# Created on: 01/09/2022
#
# Tensorflow Object Detection - Image Compressor
def main() -> int:
    from os import listdir
    from PIL import Image

    for file in listdir():
        Image.open(file).save(file, optimize=True, quality=20)

    return 0

if __name__ == "__main__":
    exit(main())
