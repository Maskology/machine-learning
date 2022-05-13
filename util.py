import os
import humanize
import shutil
import re


def remove_existing_dir(dir):
    if os.path.exists(dir) and os.path.isdir(dir):
        shutil.rmtree(dir)


def remove_existing_file(path):
    if os.path.exists(path) and os.path.isfile(path):
        os.remove(path)


def create_dir_if_not_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)


def get_file_size(path):
    return humanize.naturalsize(os.path.getsize(path))


def copy(src, dst):
    filename = os.path.basename(dst)
    remove_existing_dir(dst)
    remove_existing_file(dst)
    create_dir_if_not_exists(re.sub(filename, "", dst))

    if os.path.isdir(src):
        shutil.copytree(src, dst)
    elif os.path.isfile(src):
        shutil.copy(src, dst)
    else:
        print(f"{src} is neither directory nor file")
