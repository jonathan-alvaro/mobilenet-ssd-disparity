import pathlib


def copy_files_by_pattern(old_dir, new_dir, pattern):
    """
    Moves all files that fulfills pattern from old_dir to new_dir
    """
    old_dir = pathlib.Path(old_dir)
    matched_files = old_dir.glob(pattern)

    new_dir = pathlib.Path(new_dir)
    if not new_dir.is_dir():
        new_dir.mkdir(parents=True, exist_ok=True)

    for fp in matched_files:
        new_fp = pathlib.Path(new_dir, fp.name)
        new_fp.write_bytes(fp.read_bytes())


copy_files_by_pattern('cityscapes/train', 'train/colored_images', '**/*color*')
copy_files_by_pattern('cityscapes/test', 'test/colored_images', '**/*color*')
copy_files_by_pattern('cityscapes/val', 'val/colored_images', '**/*color*')
copy_files_by_pattern('cityscapes/train', 'train/instanceid', '**/*instanceIds*')
copy_files_by_pattern('cityscapes/test', 'test/instanceid', '**/*instanceIds*')
copy_files_by_pattern('cityscapes/val', 'val/instanceid', '**/*instanceIds*')
copy_files_by_pattern('cityscapes/train', 'train/labelid', '**/*labelIds*')
copy_files_by_pattern('cityscapes/test', 'test/labelid', '**/*labelIds*')
copy_files_by_pattern('cityscapes/val', 'val/labelid', '**/*labelIds*')
copy_files_by_pattern('bounding_boxes/train', 'train/bounding_boxes', '**/*boxes*')
copy_files_by_pattern('bounding_boxes/test', 'test/bounding_boxes', '**/*boxes*')
copy_files_by_pattern('bounding_boxes/val', 'val/bounding_boxes', '**/*boxes*')