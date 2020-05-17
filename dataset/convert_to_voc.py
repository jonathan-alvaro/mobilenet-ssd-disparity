import json
import os
import pathlib
import shutil
import xml.etree.ElementTree as ET
from typing import List
from itertools import repeat

from PIL import Image

from voc_style_labeling import wanted_labels, label_to_category_map


def discover_image_fps(image_dir: str) -> List[pathlib.Path]:
    """
    Discover all images within image_dir and returns them as pathlib.Path objects
    """
    if not os.path.isdir(image_dir):
        raise OSError("Invalid directory")

    image_dir = pathlib.Path(image_dir)
    image_fps = image_dir.glob("./*.png")

    return list(image_fps)


def extract_image_ids_from_fps(image_fps: List[pathlib.Path]) -> List[str]:
    """
    Extracts image id for each image file path in the given list
    """
    filenames = map(lambda fp: fp.stem, image_fps)
    # Image id is the name of the file before the final underscore
    image_id_components = map(lambda filename: (filename.split('_'))[:-1], filenames)
    image_ids = ['_'.join(components) for components in image_id_components]

    return list(image_ids)


def find_matching_annotation(annotation_dir: str, image_ids: List[str],
                             suffix: str = '_gtFine_boxes.json') -> List[pathlib.Path]:
    """
    Finds file paths for annotation files of each given image id
    """
    annotation_dir = pathlib.Path(annotation_dir)

    annotation_fps = [annotation_dir.joinpath(file_id + suffix) for file_id in image_ids]
    if not all(map(lambda fp: fp.is_file(), annotation_fps)):
        raise ValueError("There exists an image that doesn't have an annotation file")

    return annotation_fps


def clean_invalid_images(image_fps: List[pathlib.Path],
                         image_ids: List[str],
                         annotation_fps: List[pathlib.Path]):
    """
    Returns indices of invalid images (images with no objects of desired classes
    """
    new_image_fps = []
    new_image_ids = []
    new_annotation_fps = []

    for i, fp in enumerate(annotation_fps):
        image_objs = set()

        with fp.open() as f:
            objects = json.load(f)['objects']

        for item in objects:
            if item['label'] in wanted_labels:
                image_objs.add(item['label'])

        if len(image_objs) <= 0:
            continue
        else:
            new_image_fps.append(image_fps[i])
            new_image_ids.append(image_ids[i])
            new_annotation_fps.append(annotation_fps[i])

    return new_image_fps, new_image_ids, new_annotation_fps


def move_files(file_fps: List[pathlib.Path], new_dir: str, start_index=0) -> List[pathlib.Path]:
    """
    Moves all files into the new directory, files will be renamed into numbers
    example: 00001, 00002, etc.

    If there already exists another file with the same targeted name, it will be skipped

    Returns pathlib.Path instances of the new files
    """
    new_dir = pathlib.Path(new_dir)
    if not new_dir.is_dir():
        print("Directory {} does not exist".format(new_dir))
        print("Creating directory...")
        new_dir.mkdir(parents=True)

    file_exts = [fp.suffix for fp in file_fps]
    new_file_names = ["{:0>#5}".format(i + start_index) for i in range(len(file_fps))]

    new_fp_components = zip(repeat(new_dir, len(new_file_names)), new_file_names, file_exts)
    new_file_fps = [parent.joinpath(name + ext) for parent, name, ext in new_fp_components]

    for i, old_fp in enumerate(file_fps):
        new_fp = new_file_fps[i]
        if new_fp.exists():
            continue
        shutil.copy(old_fp, new_fp)

    return new_file_fps


def build_xml_tree(annotation_fp: pathlib.Path):
    """
    Build and writes the XML tree for annotation in given file path into another file in the same directory
    with the same name but an xml extension
    """
    filename = str(annotation_fp.stem) + '.jpg'
    with annotation_fp.open() as f:
        annotation = json.load(f)

    root = ET.Element('annotation')
    tree = ET.ElementTree(element=root)

    ET.SubElement(root, 'filename').text = filename
    ET.SubElement(root, 'folder').text = 'CityScapes'

    source = ET.Element('source')
    ET.SubElement(source, 'database').text = 'CityScapes'
    ET.SubElement(source, 'annotation').text = 'CityScapes'

    dimensions = ET.SubElement(root, 'size')
    ET.SubElement(dimensions, 'width').text = str(annotation['imgWidth'])
    ET.SubElement(dimensions, 'height').text = str(annotation['imgHeight'])
    ET.SubElement(dimensions, 'depth').text = '3'

    ET.SubElement(root, 'segmented').text = '0'

    for obj in annotation['objects']:
        if obj['label'] not in wanted_labels:
            continue
        xml_obj = ET.SubElement(root, 'object')
        ET.SubElement(xml_obj, 'name').text = obj['label']
        ET.SubElement(xml_obj, 'pose').text = 'Unspecified'
        ET.SubElement(xml_obj, 'truncated').text = '0'
        ET.SubElement(xml_obj, 'difficult').text = '0'

        bbox_corners = obj['bounding_box']
        bbox_corners = [loc if loc >= 0 else 0 for loc in bbox_corners]  # Clamp locations to within the image

        bounding_box = ET.SubElement(xml_obj, 'bndbox')
        ET.SubElement(bounding_box, 'xmin').text = str(bbox_corners[0])
        ET.SubElement(bounding_box, 'ymin').text = str(bbox_corners[1])
        ET.SubElement(bounding_box, 'xmax').text = str(bbox_corners[2])
        ET.SubElement(bounding_box, 'ymax').text = str(bbox_corners[3])

    if len(root.findall("object")) <= 0:
        print(str(annotation_fp.parent.joinpath(annotation_fp.stem + '.xml')))
        raise ValueError

    tree.write(str(annotation_fp.parent.joinpath(annotation_fp.stem + '.xml')))


def convert_annotations_to_xml(annotation_fps: List[pathlib.Path]):
    """
    Converts annotation files from JSON into VOC XML format

    XML files will be created within the same directory
    """
    for fp in annotation_fps:
        build_xml_tree(fp)


def convert_png_to_jpg(image_fps: List[pathlib.Path], new_dir: str):
    """
    Convert png cityscapes images to jpg images
    """
    new_dir = pathlib.Path(new_dir)
    if not new_dir.is_dir():
        print("Directory {} does not exist".format(new_dir))
        print("Creating directory...")
        new_dir.mkdir(parents=True)

    jpg_filenames = [fp.stem + '.jpg' for fp in image_fps]
    jpg_fps = [new_dir.joinpath(name) for name in jpg_filenames]

    for i, fp in enumerate(jpg_fps):
        png_img = Image.open(image_fps[i])
        png_img.convert("RGB").save(fp)


def create_main_image_sets(image_ids: List[str], set_file: str):
    set_file = pathlib.Path(set_file)
    if not set_file.parent.exists():
        set_file.parent.mkdir(parents=True)

    with open(str(set_file), 'w') as f:
        f.write('\n'.join(image_ids))


train_image_fps = sorted(discover_image_fps('train/images'))
val_image_fps = sorted(discover_image_fps('val/images'))
test_image_fps = sorted(discover_image_fps('val/images'))

train_image_ids = extract_image_ids_from_fps(train_image_fps)
val_image_ids = extract_image_ids_from_fps(val_image_fps)
test_image_ids = extract_image_ids_from_fps(test_image_fps)

train_annotation_fps = find_matching_annotation('train/bounding_boxes', train_image_ids)
val_annotation_fps = find_matching_annotation('val/bounding_boxes', val_image_ids)
test_annotation_fps = find_matching_annotation('val/bounding_boxes', test_image_ids)

train_image_fps, _, train_annotation_fps = clean_invalid_images(
    train_image_fps, train_image_ids, train_annotation_fps
)
val_image_fps, _, val_annotation_fps = clean_invalid_images(
    val_image_fps, val_image_ids, val_annotation_fps
)
test_image_fps, _, test_annotation_fps = clean_invalid_images(
    test_image_fps, test_image_ids, test_annotation_fps
)

train_image_fps = move_files(train_image_fps, 'voc_style_train/PNGImages')
val_image_fps = move_files(val_image_fps, 'voc_style_train/PNGImages', start_index=len(train_image_fps))
test_image_fps = move_files(test_image_fps, 'voc_style_test/PNGImages')

new_train_annotation_fps = move_files(train_annotation_fps, 'voc_style_train/Annotations')
new_val_annotation_fps = move_files(val_annotation_fps, 'voc_style_train/Annotations',
                                    start_index=len(new_train_annotation_fps))
new_test_annotation_fps = move_files(test_annotation_fps, 'voc_style_test/Annotations')

convert_annotations_to_xml(new_train_annotation_fps)
convert_annotations_to_xml(new_val_annotation_fps)
convert_annotations_to_xml(new_test_annotation_fps)

trainval_png_fps = sorted(discover_image_fps('voc_style_train/PNGImages'))
convert_png_to_jpg(trainval_png_fps, 'voc_style_train/JPEGImages')
test_png_fps = sorted(discover_image_fps('voc_style_test/PNGImages'))
convert_png_to_jpg(test_png_fps, 'voc_style_test/JPEGImages')

train_image_ids = [fp.stem for fp in train_image_fps]
val_image_ids = [fp.stem for fp in val_image_fps]
test_image_ids = [fp.stem for fp in test_image_fps]

create_main_image_sets(train_image_ids, 'voc_style_train/ImageSets/train.txt')
create_main_image_sets(val_image_ids, 'voc_style_train/ImageSets/val.txt')
create_main_image_sets(train_image_ids + val_image_ids, 'voc_style_train/ImageSets/trainval.txt')
create_main_image_sets(test_image_ids, 'voc_style_test/ImageSets/test.txt')
