# this file built on solution from kaggle team o_O: https://github.com/sveitser/kaggle_diabetic
import os
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageFilter
from multiprocessing import Process


src = 'path/to/your/dataset/folder'
tgt = 'path/to/your/processed_dataset/folder'


def main():
    jobs = []
    for root, _, imgs in os.walk(src):
        for img in tqdm(imgs):
            src_path = os.path.join(root, img)
            tgt_dir = root.replace(src, tgt)
            if not os.path.exists(tgt_dir):
                os.makedirs(tgt_dir)
            tgt_path = os.path.join(tgt_dir, img)
            jobs.append((src_path, tgt_path, 512))

    procs = []
    job_size = len(jobs) // 8
    for i in range(8):
        if i < 7:
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


def convert_square(fname, crop_size):
    img = Image.open(fname)
    bbox = square_bbox(img)
    cropped = img.crop(bbox)
    # resized = img.resize([crop_size, crop_size])
    return cropped


def get_convert_fname(fname, extension, directory, convert_directory):
    return fname.replace('JPG', extension).replace(directory,
                                                   convert_directory)


def process(args):
    fun, arg = args
    directory, convert_directory, fname, crop_size, extension = arg
    convert_fname = get_convert_fname(fname, extension, directory,
                                      convert_directory)
    if not os.path.exists(convert_fname):
        img = fun(fname, crop_size)
        save(img, convert_fname)


def save(img, fname):
    img.save(fname, quality=100, subsampling=0)


if __name__ == "__main__":
    main()
