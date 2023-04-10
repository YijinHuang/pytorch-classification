# ==========================================================================
# Base on https://github.com/sveitser/kaggle_diabetic/blob/master/convert.py
# ==========================================================================
import os
import random
import argparse
import numpy as np

from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageFilter
from multiprocessing import Process


parser = argparse.ArgumentParser()
parser.add_argument('--image-folder', type=str, help='path to image folder')
parser.add_argument('--output-folder', type=str, help='path to output folder')
parser.add_argument('--crop-size', type=int, default=512, help='crop size of image')
parser.add_argument('-n', '--num-processes', type=int, default=8, help='number of processes to use')


def main():
    args = parser.parse_args()
    image_folder = Path(args.image_folder)
    output_folder = Path(args.output_folder)

    jobs = []
    for root, _, imgs in os.walk(args.image_folder):
        root = Path(root)
        subfolders = root.relative_to(image_folder)
        output_folder = output_folder.joinpath(subfolders)
        output_folder.mkdir(parents=True, exist_ok=True)

        for img in tqdm(imgs):
            src_path = root.joinpath(img)
            tgt_path = output_folder.joinpath(img)
            jobs.append((src_path, tgt_path, args.crop_size))
    random.shuffle(jobs)

    procs = []
    job_size = len(jobs) // args.num_processes
    for i in range(args.num_processes):
        if i < args.num_processes - 1:
            procs.append(Process(target=convert_list, args=(i, jobs[i * job_size:(i + 1) * job_size])))
        else:
            procs.append(Process(target=convert_list, args=(i, jobs[i * job_size:])))

    for p in procs:
        p.start()

    for p in procs:
        p.join()


def convert_list(i, jobs):
    for j, job in enumerate(jobs):
        if j % 100 == 0:
            print('worker{} has finished {}.'.format(i, j))
        convert(*job)


def convert(fname, tgt_path, crop_size):
    img = Image.open(fname)

    blurred = img.filter(ImageFilter.BLUR)
    ba = np.array(blurred)
    h, w, _ = ba.shape

    if w > 1.2 * h:
        left_max = ba[:, : w // 32, :].max(axis=(0, 1)).astype(int)
        right_max = ba[:, - w // 32:, :].max(axis=(0, 1)).astype(int)
        max_bg = np.maximum(left_max, right_max)

        foreground = (ba > max_bg + 10).astype(np.uint8)
        bbox = Image.fromarray(foreground).getbbox()

        if bbox is None:
            print('bbox none for {} (???)'.format(fname))
        else:
            left, upper, right, lower = bbox
            # if we selected less than 80% of the original
            # height, just crop the square
            if right - left < 0.8 * h or lower - upper < 0.8 * h:
                print('bbox too small for {}'.format(fname))
                bbox = None
    else:
        bbox = None

    if bbox is None:
        bbox = square_bbox(img)

    cropped = img.crop(bbox)
    cropped = cropped.resize([crop_size, crop_size], Image.ANTIALIAS)
    save(cropped, tgt_path)


def square_bbox(img):
    w, h = img.size
    left = max((w - h) // 2, 0)
    upper = 0
    right = min(w - (w - h) // 2, w)
    lower = h
    return (left, upper, right, lower)


def save(img, fname):
    img.save(fname, quality=100, subsampling=0)


if __name__ == "__main__":
    main()