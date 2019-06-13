# -*- coding: utf-8 -*-
import argparse
import os
import xml.etree.cElementTree as ET

import cv2
import numpy as np
import pyvips

POINT_DISTANCE = 3

format_to_dtype = {
    'uchar': np.uint8,
    'char': np.int8,
    'ushort': np.uint16,
    'short': np.int16,
    'uint': np.uint32,
    'int': np.int32,
    'float': np.float32,
    'double': np.float64,
    'complex': np.complex64,
    'dpcomplex': np.complex128,
}


def vips2numpy(vi):
    return np.ndarray(buffer=vi.write_to_memory(),
                      dtype=format_to_dtype[vi.format],
                      shape=[vi.height, vi.width, vi.bands])


def scan_xml_files(xml_directory):
    xml_directory = os.path.join(xml_directory)
    if not os.path.isdir(xml_directory):
        raise Exception('xml_directory error')
    file_list = []
    for root, dirs, files in os.walk(xml_directory):
        for file in files:
            if file.split('.')[-1].lower() == 'xml':
                file_list.append(file)
    return file_list


def trim_coordinates(max_cnt_list):
    relative_point = []
    need_delete_index_list = []
    for index, value in enumerate(max_cnt_list):
        if index == 0:
            relative_point = value
        try:
            border_point = max_cnt_list[index + 1]
            distance = pow(pow(border_point[0] - relative_point[0], 2) + pow((border_point[1] - relative_point[1]), 2),
                           0.5)
            if distance <= POINT_DISTANCE:
                need_delete_index_list.append(index + 1)
            relative_point = max_cnt_list[index + 1]
        except IndexError:
            continue
    count = 0
    for i in need_delete_index_list:
        max_cnt_list.pop(i - count)
        count += 1
    return max_cnt_list


def check_contours(xml_path, image_path):
    root = ET.parse(xml_path)
    objects = root.findall('object')
    label_list = []
    coordinates_list = []
    for obj in objects:
        label_name = obj.find('name').text
        label_list.append(label_name)
        bbox_doc = obj.find('bndbox')
        x1 = int(bbox_doc.find('xmin').text)
        y1 = int(bbox_doc.find('ymin').text)
        x2 = int(bbox_doc.find('xmax').text)
        y2 = int(bbox_doc.find('ymax').text)

        try:
            pyvips_img = pyvips.Image.new_from_file(image_path)
        except WindowsError:
            basename = os.path.basename(image_path)
            raise Exception('No such image {}'.format(basename))

        patch_im = pyvips_img.extract_area(x1, y1, (x2 - x1), (y2 - y1))
        im_array = vips2numpy(patch_im)[:, :, :3]
        bgr_img = cv2.cvtColor(im_array, cv2.COLOR_RGB2BGR)

        imgray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

        roiImg2 = cv2.medianBlur(imgray, 3)  # 决定检测的灵敏度
        ret, thresh1 = cv2.threshold(roiImg2, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # 二值化

        kernel = np.ones((12, 12), np.uint8)

        dilation = cv2.dilate(thresh1, kernel, iterations=1)
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        totalContours = len(contours)
        max_area = 0
        max_cnt = 0
        for cntNo in range(0, totalContours):
            cnt = contours[cntNo]
            area = cv2.contourArea(cnt)
            if area > max_area:
                max_area = area
                max_cnt = cnt
        # 筛选
        translate_max_cnt = max_cnt.reshape((-1, 2))
        max_cnt_list = translate_max_cnt.tolist()

        max_cnt_list = trim_coordinates(max_cnt_list)

        # 修正坐标
        coordinates_line = ''
        for i in max_cnt_list:
            list = [i[0] + x1, i[1] + y1]
            coordinates_line = coordinates_line + str(list[0]) + ',' + str(list[1]) + ';'
        coordinates_line = coordinates_line[:-1]
        coordinates_list.append(coordinates_line)
    return label_list, coordinates_list


def write_result_txt_file(txt_file_path, label_list, coordinates_list):
    with open(txt_file_path, 'w') as result_file:
        for index, label in enumerate(label_list):
            result_file.writelines(label + '\n')
            result_file.writelines(coordinates_list[index] + '\n')


def parse_arg():
    parse = argparse.ArgumentParser()
    parse.add_argument('input_xml_directory', type=str, help='input_xml_directory')
    parse.add_argument('input_image_directory', type=str, help='input_image_directory')
    parse.add_argument('output_txt_directory', type=str, help='output_txt_directory')
    args = parse.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arg()
    xml_directory = args.input_xml_directory
    image_directory = args.input_image_directory
    output_txt_directory = args.output_txt_directory
    xml_file_list = scan_xml_files(xml_directory)
    for xml_file in xml_file_list:
        file_name = xml_file.split('.')[0]
        xml_path = os.path.join(xml_directory, xml_file)
        image_path = os.path.join(image_directory, file_name + '.tif')

        if not os.path.isdir(output_txt_directory):
            os.makedirs(output_txt_directory)
        txt_file_path = os.path.join(output_txt_directory, file_name + '.txt')

        label_list, coordinates_list = check_contours(xml_path, image_path)

        write_result_txt_file(txt_file_path, label_list, coordinates_list)
