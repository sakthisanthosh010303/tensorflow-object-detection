# Author: Sakthi Santhosh
# Created on: 01/09/2022
#
# Tensorflow Object Detection - Object Detection
def main(argv) -> int:
    if len(argv) < 1:
        print("Error: Program called with no data.")
        return 1

    from os import path
    from cv2 import(
        COLOR_BGR2RGB,
        cvtColor,
        imread
    )
    from matplotlib import pyplot
    from numpy import array, expand_dims, int64
    from object_detection.builders import model_builder
    from object_detection.utils import (
        config_util,
        label_map_util,
        visualization_utils
    )
    import tensorflow

    DETECTION_THRESHOLD = 0.5

    # Load pipeline config and build a detection model.
    configs = config_util.get_configs_from_pipeline_file("./models/custom/pipeline.config")
    detection_model = model_builder.build(model_config=configs["model"], is_training=False)
    category_index = label_map_util.create_category_index_from_labelmap("./workspace/label_map.pbtxt")


    # Restore checkpoint.
    checkpoint = tensorflow.compat.v2.train.Checkpoint(model=detection_model)
    checkpoint.restore("./models/custom/ckpt-5").expect_partial()

    @tensorflow.function
    def detect_fn(image):
        image, shapes = detection_model.preprocess(image)
        prediction_dict = detection_model.predict(image, shapes)
        detections = detection_model.postprocess(prediction_dict, shapes)
        return detections


    for index, file in enumerate(argv, start=1):
        count = 0
        image_handle = imread(file)
        image_array = array(image_handle)
        image_expanded = expand_dims(image_array, axis=0)

        input_tensor = tensorflow.convert_to_tensor(image_expanded, dtype=tensorflow.float32)
        detection = detect_fn(input_tensor)

        detection_count = int(detection.pop("num_detections"))
        detection = {key: value[0, :detection_count].numpy() for key, value in detection.items()}
        detection["num_detections"] = detection_count

        detection["detection_classes"] = detection["detection_classes"].astype(int64)

        label_id_offset = 1
        image_array_with_detections = image_array.copy()

        visualization_utils.visualize_boxes_and_labels_on_image_array(
            image_array_with_detections,
            detection["detection_boxes"],
            detection["detection_classes"] + label_id_offset,
            detection["detection_scores"],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=5,
            min_score_thresh=.8,
            line_thickness=20,
            agnostic_mode=False
        )

        pyplot.imshow(cvtColor(image_array_with_detections, COLOR_BGR2RGB))
        pyplot.savefig("./workspace/output/temp_image%d.jpg"%(index))
        print("\nOutput: temp_image%d.jpg"%(index))

        # Count number of objects detected. Criteria: Prediction > 0.8
        for score in detection["detection_scores"]:
            if score > DETECTION_THRESHOLD:
                count += 1
        print("Count:", count)
    return 0

if __name__ == "__main__":
    from sys import argv

    exit(main(argv[1:]))
