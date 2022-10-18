# Author: Sakthi Santhosh
# Created on: 08/09/2022
#
# CLA: image_path, xml_path, label_map and output_path
#
# Tensorflow Object Detection - Record Generator Script
def main(argv) -> int:
    if len(argv) < 4:
        print("Error: Program called with incomplete data.")
        return 1

    from collections import namedtuple
    from io import BytesIO
    from object_detection.utils import (
        dataset_util,
        label_map_util
    )
    from os import (
        listdir,
        path
    )
    from PIL import Image
    from pandas import DataFrame
    from tensorflow.compat.v1 import (
        gfile,
        python_io,
        train
    )
    from xml.etree import ElementTree

    # Convert XML to CSV.
    COLUMNS = [
        "filename",
        "width",
        "height",
        "class",
        "xmin",
        "ymin",
        "xmax",
        "ymax"
    ]
    label_map = label_map_util.get_label_map_dict(
        label_map_util.load_labelmap(argv[2])
    )
    xmls = []

    for file in listdir(argv[1]):
        root = ElementTree.parse(path.join(argv[1], file)).getroot()
        for member in root.findall("object"):
            xmls.append((
                root.find("filename").text,
                int(root.find("size")[0].text),
                int(root.find("size")[1].text),
                member[0].text,
                int(member[4][0].text),
                int(member[4][1].text),
                int(member[4][2].text),
                int(member[4][3].text)
            ))
    dataframe = DataFrame(xmls, columns=COLUMNS)
    dataframe_group = dataframe.groupby("filename")
    data = namedtuple("data", ["filename", "object"])
    groups = [
        data(file, dataframe_group.get_group(metadata)) for file, metadata in zip(
            dataframe_group.groups.keys(), dataframe_group.groups
    )]

    # Generate TFRecord.
    write_handle = python_io.TFRecordWriter(argv[3])
    for group in groups:
        with gfile.GFile(path.join(argv[0], group.filename), "rb") as file_handle:
            image_bytes = file_handle.read()
            image = Image.open(BytesIO(image_bytes))
            width, height = image.size
            filename = group.filename.encode("utf-8")
            xmins, ymins, xmaxs, ymaxs, class_texts, classes = [], [], [], [], [], []

            for index, row in group.object.iterrows():
                xmins.append(row["xmin"] / width)
                xmaxs.append(row["xmax"] / width)
                ymins.append(row["ymin"] / height)
                ymaxs.append(row["ymax"] / height)
                class_texts.append(row["class"].encode("utf8"))
                classes.append(label_map[row["class"]])

        tensorflow_example = train.Example(
            features=train.Features(
                feature={
                    "image/height": dataset_util.int64_feature(height),
                    "image/width": dataset_util.int64_feature(width),
                    "image/filename": dataset_util.bytes_feature(filename),
                    "image/source_id": dataset_util.bytes_feature(filename),
                    "image/encoded": dataset_util.bytes_feature(image_bytes),
                    'image/format': dataset_util.bytes_feature(b"jpg"),
                    'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
                    'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
                    'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
                    'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
                    'image/object/class/text': dataset_util.bytes_list_feature(class_texts),
                    'image/object/class/label': dataset_util.int64_list_feature(classes)
        }))

        write_handle.write(tensorflow_example.SerializeToString())
    write_handle.close()

    print("Output:", argv[3])
    return 0

if __name__ == "__main__":
    from sys import argv

    exit(main(argv[1:]))
