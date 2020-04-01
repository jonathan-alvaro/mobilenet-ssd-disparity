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


copy_files_by_pattern('cityscapes/train', 'colored_images/train', '**/*color*')
copy_files_by_pattern('cityscapes/test', 'colored_images/test', '**/*color*')
copy_files_by_pattern('cityscapes/val', 'colored_images/val', '**/*color*')
copy_files_by_pattern('cityscapes/train', 'instanceid/train', '**/*instanceIds*')
copy_files_by_pattern('cityscapes/test', 'instanceid/test', '**/*instanceIds*')
copy_files_by_pattern('cityscapes/val', 'instanceid/val', '**/*instanceIds*')
copy_files_by_pattern('cityscapes/train', 'labelid/train', '**/*labelIds*')
copy_files_by_pattern('cityscapes/test', 'labelid/test', '**/*labelIds*')
copy_files_by_pattern('cityscapes/val', 'labelid/val', '**/*labelIds*')