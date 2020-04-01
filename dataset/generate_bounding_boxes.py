import json
import pathlib


def list_polygons_files(directory):
    dir_path = pathlib.Path(directory)
    polygon_paths = dir_path.glob('**/*polygons*')

    return list(polygon_paths)


def polygon_to_bounding_boxes(polygon):
    min_x = 9999
    min_y = 9999
    max_x = -1
    max_y = -1

    for vertex in polygon:
        x, y = vertex
        if x < min_x:
            min_x = x
        if x > max_x:
            max_x = x
        if y < min_y:
            min_y = y
        if y > max_y:
            max_y = y

    return min_x, min_y, max_x, max_y


def extract_bounding_boxes(objects):
    for obj in objects:
        object_polygon = obj.pop('polygon')
        min_x, min_y, max_x, max_y = polygon_to_bounding_boxes(object_polygon)
        obj['bounding_box'] = (min_x, min_y, max_x, max_y)

    return objects


def convert_polygon_file(original_file_path, new_file_path):
    with open(original_file_path) as f:
        file_json = json.load(f)

    new_objects = extract_bounding_boxes(file_json['objects'])
    file_json['objects'] = new_objects

    json.dump(file_json, open(new_file_path, 'w'))


def reformat_polygons(original_dir, new_dir):
    polygon_files = list_polygons_files(original_dir)

    new_dir = pathlib.Path(new_dir)
    if not new_dir.is_dir():
        new_dir.mkdir(parents=True, exist_ok=True)

    for filepath in polygon_files:
        new_filename = filepath.name.replace('polygons', 'boxes')
        new_filepath = pathlib.Path(new_dir, new_filename)

        convert_polygon_file(filepath, new_filepath)


reformat_polygons('cityscapes/train', 'bounding_boxes/train')
reformat_polygons('cityscapes/test', 'bounding_boxes/test')
reformat_polygons('cityscapes/val', 'bounding_boxes/val')
