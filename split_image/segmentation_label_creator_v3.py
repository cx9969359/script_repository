import argparse
import base64
import copy
import json
import os
import re
import sys
import time

import cv2
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('input_json_path', help='path to json files')
parser.add_argument('input_palette', help='path to palette file')
parser.add_argument('output_png_path', help='output path for png files')
parser.add_argument('init_type', type=int,
                    help='the init value for png files, 0 means using value 0, 1 means using the value of ignore label')

ignore_label_index = 255


def get_label_indexes(input_json, label_string):
    all_shapes = input_json['shapes']
    all_shapes_len = len(all_shapes)
    index_list = []
    # for i in range(len(pointer_array_list))
    #     if pointer_array_list[i]['label'] == label_string:
    #           index_list.append(i)
    # return index_list
    for num in range(0, all_shapes_len):
        if all_shapes[num]['label'] == label_string:
            index_list.append(num)
    return index_list


def draw_single_contour_to_image(input_image, contour, color_filled, color_border=None):
    contour = np.reshape(contour, (-1, 1, 2))
    contour = contour.astype(np.int32)

    contours = []
    contours.append(contour)
    cv2.drawContours(input_image, contours, -1, (color_filled), -1)

    if color_border:
        cv2.drawContours(input_image, contours, -1, (color_border), 5)
    return input_image


def create_contour_image(img_np, all_shapes, label_shape_dict, label_list, init_type):
    if init_type == 0:
        img_mat = np.full((img_np.shape[0], img_np.shape[1], 1), float(0))
    else:
        img_mat = np.full((img_np.shape[0], img_np.shape[1], 1), float(ignore_label_index))

    for tmp_color_index in range(len(label_list) - 1):
        for tmp_label_index in range(len(label_list[tmp_color_index])):
            for tmp_shape_index in label_shape_dict[label_list[tmp_color_index][tmp_label_index]]:
                contour = np.asarray(all_shapes[tmp_shape_index]['points'])
                draw_single_contour_to_image(img_mat, contour, (tmp_color_index), (ignore_label_index))

    return img_mat


def write_single_image(input_json_file, dest_path, palette_list, label_list, init_type):
    with open(input_json_file) as jsonfile:
        json_data = json.load(jsonfile)

    label_shape_dict = {}
    # len(label_list) = 256
    # for i in range(256):
    #     for j in range(len(i)):
    #          label_shape_dict[label_list[i][j]] = get_label_index(json_data, label_list[[i][j]])
    for tmp_color_index in range(len(label_list)):
        for tmp_label_index in range(len(label_list[tmp_color_index])):
            label_shape_dict[label_list[tmp_color_index][tmp_label_index]] = get_label_indexes(json_data, label_list[
                tmp_color_index][tmp_label_index])

    # vips拿到image的nparr，转化成img_np
    all_shapes = json_data['shapes']
    base64data = json_data['imageData']
    imgData = base64.b64decode(base64data)

    nparr = np.fromstring(imgData, np.uint8)
    img_np = cv2.imdecode(nparr, 1)
    output_mat = create_contour_image(img_np, all_shapes, label_shape_dict, label_list, init_type)

    check_mat = np.full(output_mat.shape, float(ignore_label_index))

    if np.array_equal(output_mat, check_mat):
        print('all label ignored for file {}'.format(input_json_file))
    else:
        pil_mat = np.squeeze(output_mat, axis=2)
        pil_img = Image.fromarray(pil_mat)
        pil_img = pil_img.convert('P')
        pil_img.putpalette(palette_list)
        pil_img.save(dest_path)


def read_palette_file(palette_file_path):
    palette_list = [0] * (256 * 3)
    label_list = [''] * 256
    counter = 0
    with open(palette_file_path) as f:
        for line in f:
            line = line.rstrip()
            palette_list[counter:counter + 3] = map(int, line.split(' ')[:3])
            label_list[int(counter / 3)] = line.split(' ')[3:]
            counter += 3
            if counter >= (256 * 3):
                break

    if counter < 256 * 3:
        palette_list[-3:] = palette_list[counter - 3:counter]
        palette_list[counter - 3:counter] = [0, 0, 0]
        label_list[-1] = label_list[int((counter - 3) / 3)]
        label_list[int((counter - 3) / 3)] = ''

    return palette_list, label_list


def scan_files_and_create_folder(input_json_path, output_png_path, ext_list):
    file_list = []
    for root, dirs, files in os.walk(input_json_path):
        # create folder if it does not exist
        new_path = re.sub(os.path.join(input_json_path, ''), os.path.join(output_png_path, ''), os.path.join(root, ''))
        if not os.path.exists(new_path):
            os.makedirs(new_path)

        # scan all files and put it into list
        for f in files:
            if os.path.splitext(f)[1].lower() in ext_list:
                file_list.append(os.path.join(root, f))

    return file_list


def write_voc_seg_label_png_files(json_file_list, input_palette, input_json_path, output_png_path, init_type):
    palette_list, label_list = read_palette_file(input_palette)
    print(palette_list)
    print(label_list)

    for json_file in json_file_list:
        output_file = re.sub(os.path.join(input_json_path, ''), os.path.join(output_png_path, ''),
                             (os.path.splitext(json_file)[0])) + '.png'
        try:
            write_single_image(json_file, output_file, palette_list, label_list, init_type)
        except Exception as e:
            print('get error for file: ' + json_file)


def main():
    # input_json_path = args.input_json_path
    # input_palette = args.input_palette
    # output_png_path = args.output_png_path
    # init_type = args.init_type
    input_json_path = ''
    input_palette = 'F:/split_image/palette_folder/palette.txt'
    output_png_path = 'F:/split_image/out_put_png'
    init_type = 0

    print('start to scan files and create folders')
    json_file_list = scan_files_and_create_folder(input_json_path, output_png_path, ['.json'])
    print('done for scaning files and creating folders')

    print('start to write png files')
    write_voc_seg_label_png_files(json_file_list, input_palette, input_json_path, output_png_path, init_type)
    print('done for writing png files')


if __name__ == '__main__':
    # args = parser.parse_args()
    # main(args)
    main()
