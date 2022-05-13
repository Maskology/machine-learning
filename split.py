import os
import random
import multiprocessing
import concurrent.futures
from util import remove_existing_dir, copy


class DatasetDirectoryLabelSplit:
    def __init__(
        self,
        src_dir,
        dst_dir,
        train_split=0.8,
        validation_split=None,
        shuffle=True,
    ):
        self.src_dir = src_dir
        self.dst_dir = dst_dir

        self.train_split = train_split
        self.validation_split = validation_split
        self.shuffle = shuffle

        self.train_n = 0
        self.validation_n = 0
        self.test_n = 0

        self.label_distribution = []
        self.set_label_distribution()

        self.distribution_path = {}
        self.copy_paths = []
        self.set_copy_paths()

    def create(self):
        remove_existing_dir(self.dst_dir)
        self.create_concurrency()

    def create_concurrency(self):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(lambda args: copy(*args), self.copy_paths)

    def create_parallel(self):
        with multiprocessing.Pool() as pool:
            pool.starmap(copy, self.copy_paths)

    def set_copy_paths(self):
        for data in self.label_distribution:
            for dist, files in data["dist"].items():
                dist_dir = os.path.join(self.dst_dir, dist)
                self.distribution_path[dist] = dist_dir
                for f in files:
                    src_file = os.path.join(self.src_dir, data["label"], f)
                    dst_file = os.path.join(dist_dir, data["label"], f)
                    self.copy_paths.append((src_file, dst_file))

    def set_label_distribution(self):
        labels = os.listdir(self.src_dir)
        for label in labels:
            files = os.listdir(os.path.join(self.src_dir, label))
            if self.shuffle:
                random.shuffle(files)
            train_n, val_n, test_n, = self.get_num_train_val_test_split(
                len(files), self.train_split, self.validation_split
            )
            self.train_n += train_n
            self.validation_n += val_n
            self.test_n += test_n
            self.label_distribution.append(
                {
                    "label": label,
                    "dist": {
                        # 'all': files,
                        "train": files[:train_n],
                        "validation": files[train_n : train_n + val_n],
                        "test": files[-test_n:],
                    },
                }
            )

    def get_num_train_val_test_split(self, n, train_split, validation_split=None):
        train_n = round(n * train_split)
        rest = n - train_n
        val_n = round(n * validation_split) if validation_split else round(rest / 2)
        test_n = rest - val_n
        return train_n, val_n, test_n
