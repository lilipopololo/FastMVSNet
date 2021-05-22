
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import cv2


def pfm_png_file_name(pfm_file_path, png_file_path):
    png_file_path = {}
    for root, dirs, files in os.walk(pfm_file_path):
        for file in files:
            file = os.path.splitext(file)[0] + ".png"
            files = os.path.splitext(file)[0] + ".pfm"
            png_file_path = os.path.join(root, file)
            pfm_file_path = os.path.join(root, files)
            pfm_png(pfm_file_path, png_file_path)


def pfm_png(pfm_file_path, png_file_path):
    if os.path.isfile(pfm_file_path)==False:
        return 
        
    with open(pfm_file_path, 'rb') as pfm_file:
        header = pfm_file.readline().decode().rstrip()
        channel = 3 if header == 'PF' else 1

        dim_match = re.match(r'^(\d+)\s(\d+)\s$', pfm_file.readline().decode('utf-8'))
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception("Malformed PFM header.")

        scale = float(pfm_file.readline().decode().strip())
        if scale < 0:
            endian = '<'  # little endlian
            scale = -scale
        else:
            endian = '>'  # big endlian

        disparity = np.fromfile(pfm_file, endian + 'f')

        img = np.reshape(disparity, newshape=(height, width))
        img = np.flipud(img)
        plt.imsave(os.path.join(png_file_path), img)


def main():
    test_set = [1, 4, 9, 10, 11, 12, 13, 15, 23, 24, 29, 32, 33, 34, 48, 49, 62, 75, 77,
            110, 114, 118]
    pfm_base_dir = 'E:/dataset/dtu_training/dtu_training/mvs_training/dtu/Eval/Rectified/dtu'
    png_file_dir = pfm_base_dir
    pfm_file_dir = [ pfm_base_dir + '/scan' + str(i) for i in test_set]
    for pfm_file_dir in pfm_file_dir:
        print(pfm_file_dir)
        pfm_png_file_name(pfm_file_dir, png_file_dir)


if __name__ == '__main__':
    main()