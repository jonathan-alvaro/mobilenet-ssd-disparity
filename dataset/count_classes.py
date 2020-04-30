import json
import pathlib

def count_classes(annotation_dir: str) -> dict:
    annotation_dir = pathlib.Path(annotation_dir)

    annotation_fps = annotation_dir.glob("./*json")

    class_counts = {}

    for fp in annotation_fps:
        with fp.open() as f:
            objects = json.load(f)['objects']
        
        for item in objects:
            if item['label'] in class_counts:
                class_counts[item['label']] += 1
            else:
                class_counts[item['label']] = 1

    return class_counts

print(count_classes('train/bounding_boxes'))