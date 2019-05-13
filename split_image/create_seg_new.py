import os
import xml.etree.ElementTree as ET
from functools import partial
from multiprocessing import Pool, cpu_count

import cv2
import imutils
import numpy as np
import pyvips
import scipy.stats as st
from PIL import Image

# map vips formats to np dtypes
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


def scan_files_and_create_folder(input_file_path, output_seg_path, ext_list):
    file_list = []
    for root, dirs, files in os.walk(input_file_path):
        # create folder if it does not exist
        new_path = os.path.join(root, '').replace('\\', '/').replace(
            os.path.join(input_file_path, '').replace('\\', '/'), os.path.join(output_seg_path, '').replace('\\', '/'),
            1)
        if not os.path.exists(new_path):
            os.makedirs(new_path)

        # scan all files and put it into list
        for f in files:
            if os.path.splitext(f)[1].lower() in ext_list:
                file_list.append(os.path.splitext(os.path.join(root, f).replace('\\', '/').replace(
                    os.path.join(input_file_path, '').replace('\\', '/'), ''))[0])
    return file_list


def get_image_shape(xml_tree):
    height = int(xml_tree.find('size').find('height').text)
    width = int(xml_tree.find('size').find('width').text)
    return (height, width)


def get_regions(img_shape, crop_size, overlap):
    regions = []
    assert img_shape[1] >= crop_size and img_shape[0] >= crop_size
    assert crop_size > overlap
    h_start = 0
    while h_start < img_shape[0]:
        w_start = 0
        while w_start < img_shape[1]:
            region_x2 = min(max(0, w_start + crop_size), img_shape[1])
            region_y2 = min(max(0, h_start + crop_size), img_shape[0])
            region_x1 = min(max(0, region_x2 - crop_size), img_shape[1])
            region_y1 = min(max(0, region_y2 - crop_size), img_shape[0])
            regions.append([region_x1, region_y1, region_x2, region_y2])
            # break when region reach the end
            if w_start + crop_size >= img_shape[1]: break
            w_start += crop_size - overlap
        # break when region reach the end
        if h_start + crop_size >= img_shape[0]: break
        h_start += crop_size - overlap
    return regions


def get_regions_record(xml_tree, doing_list, ignore_list, regions, crop_size, crop_threshold=0., file_postfix='',
                       zfill_value=0):
    objects = xml_tree.findall('object')
    pool = Pool(cpu_count())
    _get_record = partial(get_record, doing_list=doing_list, ignore_list=ignore_list, objects=objects,
                          crop_threshold=crop_threshold, file_postfix=file_postfix,
                          zfill_value=zfill_value, crop_size=crop_size)
    regions_record = pool.map(_get_record, regions)
    while None in regions_record:
        regions_record.remove(None)
    return regions_record


def get_record(region, doing_list, ignore_list, objects, crop_threshold, file_postfix, zfill_value, crop_size):
    obj_index_list = []
    for index, obj in enumerate(objects):
        cls_name = obj.find('name').text.lower().strip()
        if len(ignore_list) != 0 and cls_name in ignore_list:
            continue
        elif len(doing_list) != 0 and (cls_name not in doing_list):
            continue
        obj_index_list.append(index)

    x1, y1, x2, y2 = region
    keep = []
    ls = 0
    rs = 0
    us = 0
    ds = 0
    for index in obj_index_list:
        bbox = objects[index].find('bndbox')
        bx1 = float(bbox.find('xmin').text)
        by1 = float(bbox.find('ymin').text)
        bx2 = float(bbox.find('xmax').text)
        by2 = float(bbox.find('ymax').text)

        ix1 = max(bx1, x1)
        iy1 = max(by1, y1)
        ix2 = min(bx2, x2)
        iy2 = min(by2, y2)

        w = max(0, ix2 - ix1)
        h = max(0, iy2 - iy1)
        inter = w * h

        if inter > 0:
            ls = int(max(max((x1 - bx1), 0), ls))
            rs = int(max(max((bx2 - x2), 0), rs))
            us = int(max(max((y1 - by1), 0), us))
            ds = int(max(max((by2 - y2), 0), ds))

            keep.append(index)

    # base array, fill 1 for each object
    mask_arr = np.zeros((us + crop_size + ds, ls + crop_size + rs), dtype=np.float32)

    # draw 1 for all obj in current region
    points_list = []
    for index in keep:
        dict = {}
        label = objects[index].find('name').text
        points_arr = get_all_points(objects[index])
        points_arr = correct_points(points_arr, x1 - ls, y1 - us)
        mask_arr = fill_by_points(points_arr, mask_arr)

        points = points_arr.tolist()
        dict[label] = points
        points_list.append(dict)

    if (np.sum(mask_arr[us: us + crop_size, ls:ls + crop_size]) / (crop_size * crop_size)) <= crop_threshold:
        return
    # print('generate regions_record')
    result = [x1, y1, x2, y2] + points_list
    return result


def fill_by_points(points_arr, img_arr, value=(1)):
    tmp_arr = img_arr.copy()
    pts = points_arr.copy()
    pts = pts.reshape((-1, 1, 2))
    return cv2.fillPoly(tmp_arr, [pts], value)


def get_all_points(obj):
    points = []

    for point_obj in obj.find('segmentation').findall('points'):
        points.append(int(float(point_obj.find('x').text)))
        points.append(int(float(point_obj.find('y').text)))

    return np.asarray(points, dtype=np.int32)


def get_rot_box_info(obj):
    # rot_box: [cen_x, cen_y, width, height, angle]
    rot_box = []
    rot_obj = obj.find('rot_box')
    rot_box.append(int(float(rot_obj.find('center_x').text)))
    rot_box.append(int(float(rot_obj.find('center_y').text)))
    rot_box.append(int(float(rot_obj.find('box_width').text)))
    rot_box.append(int(float(rot_obj.find('box_height').text)))
    rot_box.append(abs(float(rot_obj.find('box_angle').text)))
    return rot_box


def correct_points(points_arr, start_x, start_y):
    points_arr[0::2] = points_arr[0::2] - start_x
    points_arr[1::2] = points_arr[1::2] - start_y
    return points_arr


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


def draw_single_contour_to_image(input_image, contour, color_filled, color_border=None):
    contour = np.reshape(contour, (-1, 1, 2))
    contour = contour.astype(np.int32)

    contours = []
    contours.append(contour)
    cv2.drawContours(input_image, contours, -1, color_filled, -1)

    if color_border:
        cv2.drawContours(input_image, contours, -1, (color_border), 5)
    return input_image


def create_contour_image(img_np, ignore_label_index, label_shape_dict, label_list, init_type):
    if init_type == 0:
        img_mat = np.full((img_np.shape[0], img_np.shape[1], 1), float(0))
    else:
        img_mat = np.full((img_np.shape[0], img_np.shape[1], 1), float(ignore_label_index))
    # temple_format: label_shape_dict = {'lsil': [[point_list], [point_list], ...], 'hsil': []}
    for tmp_color_index in range(len(label_list) - 1):
        for tmp_label_index in range(len(label_list[tmp_color_index])):
            for tmp_points_list in label_shape_dict[label_list[tmp_color_index][tmp_label_index]]:
                contour = np.asarray(tmp_points_list)
                draw_single_contour_to_image(img_mat, contour, (tmp_color_index), (ignore_label_index))
    return img_mat


def get_label_indexes(file_point_array_list, label_string):
    all_shapes = file_point_array_list
    all_shapes_len = len(all_shapes)
    index_list = []
    for num in range(0, all_shapes_len):
        if all_shapes[num]['label'] == label_string:
            index_list.append(num)

    return index_list


def get_label_point_list(annotation_xml_tree, label):
    objects = annotation_xml_tree.findall('object')
    point_list = []
    for index, object in enumerate(objects):
        if object.find('name').text == label:
            points = object.find('segmentation').findall('points')
            tuple_list = []
            for p in points:
                point_tuple = (float(p.find('x').text), float(p.find('y').text))
                tuple_list.append(point_tuple)
            point_list.append(tuple_list)
    return point_list


if __name__ == '__main__':

    annotation_xml_path = '././label_xml/V201803956LSIL_2019_01_28_15_26_39.xml'
    image_path = 'F:/tif_images/thyroid/V201803956LSIL_2019_01_28_15_26_39.tif'
    output_dir = '././out_put_png'
    palette_path = './palette_folder/palette.txt'
    crop_size = 900
    overlap = 300
    ignore_label_index = 255
    doing_list = ['hsil', 'scc', 'lsil']
    ignore_list = []
    annotation_xml_tree = ET.parse(annotation_xml_path)

    palette_list, label_list = read_palette_file(palette_path)
    image_shape = get_image_shape(annotation_xml_tree)
    regions = get_regions(image_shape, crop_size, overlap)

    regions_record = get_regions_record(annotation_xml_tree, doing_list, ignore_list, regions, crop_size,
                                        crop_threshold=0., file_postfix='cropped', zfill_value=12)
    pyvips_image = pyvips.Image.new_from_file(image_path)
    count = 0
    for region in regions_record:
        patch = pyvips_image.extract_area(region[0], region[1], region[2] - region[0], region[3] - region[1])
        img_np = vips2numpy(patch)
        init_type = 0

        if init_type == 0:
            img_mat = np.full((img_np.shape[0], img_np.shape[1], 1), float(0))
        else:
            img_mat = np.full((img_np.shape[0], img_np.shape[1], 1), float(ignore_label_index))

        contour_dict = region[-1]
        for label, points in contour_dict.items():
            contour = np.array(points)
            color_filled = (128, 128, 128)
            for i in range(len(label_list) - 1):
                for j in range(len(label_list[i])):
                    if label_list[i][j] == label:
                        img_mat = draw_single_contour_to_image(img_mat, contour, (i), (ignore_label_index))

        output_mat = img_mat
        check_mat = np.full(output_mat.shape, float(ignore_label_index))

        if np.array_equal(output_mat, check_mat):
            print('all label ignored for file')
        else:
            pil_mat = np.squeeze(output_mat, axis=2)
            pil_img = Image.fromarray(pil_mat)
            pil_img = pil_img.convert('P')
            pil_img.putpalette(palette_list)
            file_name = os.path.basename(os.path.join(annotation_xml_path)).split('.')[0]
            save_path = os.path.join(output_dir, '{}-{}.png'.format(file_name, count))
            pil_img.save(save_path)
            count += 1
