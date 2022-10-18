# Author: Sakthi Santhosh
# Created on: 01/09/2022
#
# Tensorflow Object Detection - File Renamer
def main(argv) -> int:
    if len(argv) < 1:
        print("Error: Program called with no data.")
        return 1

    from os import(
        listdir,
        rename
    )

    for index, file in enumerate(listdir(), start=1):
        rename(file, "%s_image%d.jpg"%(argv[0], index))
    return 0

if __name__ == "__main__":
    from sys import argv

    exit(main(argv[1:]))
