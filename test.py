import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_NPY = "/home/wt/Desktop/changsha/songbiao_0625/training/1_mask.npy"


def load_mask(path):
    data = np.load(path)
    if data.ndim != 2:
        raise ValueError("只支持二维npy可视化，当前shape={}".format(data.shape))
    return data


def visualize_mask(mask, path):
    print("path:", path)
    print("shape:", mask.shape)
    print("dtype:", mask.dtype)
    print("min/max:", np.min(mask), np.max(mask))
    print("unique:", np.unique(mask))

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(mask.T, origin="upper", interpolation="nearest", cmap="tab20")
    ax.set_title(os.path.basename(path))
    ax.set_xlabel("x index")
    ax.set_ylabel("y index")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("mask id")
    fig.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="可视化导出的mask npy文件")
    parser.add_argument("path", nargs="?", default=DEFAULT_NPY, help="npy文件路径")
    args = parser.parse_args()

    mask = load_mask(args.path)
    visualize_mask(mask, args.path)


if __name__ == "__main__":
    main()
