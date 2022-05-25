import os
import re
import random
import multiprocessing
import concurrent.futures
import matplotlib.pyplot as plt
from PIL import Image
from util import (
    remove_existing_dir,
    remove_existing_file,
    create_dir_if_not_exists,
    get_file_size,
)


class CompressImgDataset():

    def __init__(
            self,
            src_dir,
            dst_dir,
            img_size=(128, 128),
            exclude_below=True,
            quality=10,
            overwrite_exists=False,
            limit_file=None
    ):
        self.src_dir = src_dir
        self.dst_dir = dst_dir

        self.img_size = img_size
        self.limit_file = limit_file
        self.quality = quality
        self.overwrite_exists = overwrite_exists
        self.exclude_below = exclude_below

        self.result_paths = []
        self.set_result_paths()

    def compress(self):
        if os.path.exists(self.dst_dir) and not self.overwrite_exists:
            print('Compressed dataset already exists!')
            return
        remove_existing_dir(self.dst_dir)
        self.compress_concurrency()

    def compress_concurrency(self):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(lambda args: self.compress_img(*args), self.result_paths)

    def compress_parallel(self):
        with multiprocessing.Pool() as pool:
            pool.starmap(self.compress_img, self.result_paths)

    def compress_img(self, src_path, dst_path):
        filename = os.path.basename(dst_path)
        dst_dir = re.sub(filename, '', dst_path)
        create_dir_if_not_exists(dst_dir)
        remove_existing_file(dst_path)
        img = Image.open(src_path)
        original_size = img.size
        img.thumbnail(self.img_size, Image.ANTIALIAS)

        is_excluded = original_size[0] < self.img_size[0] or original_size[1] < self.img_size[1]
        if self.exclude_below and is_excluded:
            img.save(dst_path)
            return

        img.save(dst_path, quality=self.quality, optimize=True)

    def set_result_paths(self):
        for path, subdirs, files in os.walk(self.src_dir):
            for name in files:
                if isinstance(self.limit_file, int) and len(self.result_paths) >= self.limit_file:
                    break
                src_path = os.path.join(path, name)
                dst_path = re.sub(self.src_dir, self.dst_dir, src_path)
                self.result_paths.append((src_path, dst_path))

    def show_result_samples(self, n=2, figsize=None, axis='on'):
        samples = random.sample(self.result_paths, n)
        fig, axes = plt.subplots(len(samples), 2, figsize=figsize, constrained_layout=True)
        for i, sample in enumerate(samples):
            original, compressed = sample
            sample = ((original, 'Original'), (compressed, 'Compressed'))
            for j, data in enumerate(sample):
                path, kind = data
                img = Image.open(path)
                ax = axes[i][j]
                ax.imshow(img)
                ax.set_title(f'{kind} {get_file_size(path)}')
                ax.set_xlabel(img.size)
                ax.set_yticklabels([])
                ax.set_xticklabels([])
                ax.axis(axis)
        plt.show()