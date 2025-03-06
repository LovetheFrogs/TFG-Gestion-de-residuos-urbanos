"""Creates N datasets, each one with one more node. Starts at 2 nodes."""

import create_models as cm
import shutil
import os


N = 75


if __name__ == "__main__":
    for i in range(2, N):
        cm.MIN_NODES = i
        cm.MAX_NODES = i
        cm.create_dataset()
        shutil.move(f'{os.getcwd()}/files/datasets/dataset1.txt', 
                    f'{os.getcwd()}/files/test2tonnodes/{i}n.txt')
