import argparse
import xml.etree.ElementTree
from xml.dom.minidom import Document
import os, time
import ast
from multiprocessing import Process, Queue

import cv2
import numpy as np
import pyvips
import yaml
import pickle

import mxnet as mxnet
from mxnet.module import Module

from symnet.model import load_param, check_shape
from alg_system import alg_system
from util.nms_filter import nms_in_class, nms_between_classes, filter_boxes_dict


# multiprocessing part
def infer_obj_worker(input, output):
    det_system = ""
    for gpu_index, region, patch, args in iter(input.get, 'STOP'):
        x1, y1, x2, y2 = int(region[0]), int(region[1]), int(region[2]), int(region[3])
        patch = patch[:, :, ::-1]

        # init
        if det_system == "":
            det_system = alg_system(args, gpu_index)

        # infer
        tmp_result = det_system.infer_image(patch)
        tmp_result = correct_result(tmp_result, [x1, y1, x1, y1])
        output.put(tmp_result)


def parse_args():
    parser = argparse.ArgumentParser(description='Test a Faster R-CNN network',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('yml_path', type=str, help='path to image')

    args = parser.parse_args()
    yml_file = open(args.yml_path, encoding='utf-8')

    param_dict = yaml.safe_load(yml_file)
    for item in param_dict:
        print('key', item, 'type:', type(param_dict[item]), 'value:', param_dict[item])
        parser.add_argument(item, type=type(param_dict[item]), default=param_dict[item], help='');
    args = parser.parse_args()
    parser.add_argument('--rcnn_num_classes', type=int, default=len(args.CLASSES_NAME), help='');
    args = parser.parse_args()
    args.img_pixel_means = ast.literal_eval(args.img_pixel_means)
    args.img_pixel_stds = ast.literal_eval(args.img_pixel_stds)
    args.rpn_anchor_scales = ast.literal_eval(args.rpn_anchor_scales)
    args.rpn_anchor_ratios = ast.literal_eval(args.rpn_anchor_ratios)
    args.rcnn_pooled_size = ast.literal_eval(args.rcnn_pooled_size)
    args.rcnn_bbox_stds = ast.literal_eval(args.rcnn_bbox_stds)
    return args


def scan_files(input_path, ext_list):
    file_list = []
    for root, dirs, files in os.walk(input_path):
        # scan all files and put it into list
        for f in files:
            if os.path.splitext(f)[1].lower() in ext_list:
                file_list.append(os.path.join(root, f).replace(os.path.join(input_path, ""), "", 1))
    return file_list


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

# map np dtypes to vips
dtype_to_format = {
    'uint8': 'uchar',
    'int8': 'char',
    'uint16': 'ushort',
    'int16': 'short',
    'uint32': 'uint',
    'int32': 'int',
    'float32': 'float',
    'float64': 'double',
    'complex64': 'complex',
    'complex128': 'dpcomplex',
}


def generate_whole_xml_contents(region_dict, img_size):
    doc = Document()
    anno = doc.createElement('annotation')
    doc.appendChild(anno)
    filename = doc.createElement('filename')
    filename.appendChild(doc.createTextNode("infer_file"))
    anno.appendChild(filename)

    size = doc.createElement('size')
    size.appendChild(doc.createElement('width')).appendChild(doc.createTextNode(str(int(img_size[0]))))
    size.appendChild(doc.createElement('height')).appendChild(doc.createTextNode(str(int(img_size[1]))))

    size.appendChild(doc.createElement('depth')).appendChild(doc.createTextNode(str(3)))
    anno.appendChild(size)

    for label_name in region_dict:
        for box in region_dict[label_name]:
            obj = doc.createElement('object')
            anno.appendChild(obj)
            obj.appendChild(doc.createElement('name')).appendChild(doc.createTextNode(label_name))
            diff = doc.createElement('difficult');
            diff.appendChild(doc.createTextNode(str(0)))
            trun = doc.createElement('truncated');
            trun.appendChild(doc.createTextNode(str(0)))
            obj.appendChild(diff)
            obj.appendChild(trun)

            bndbox = doc.createElement('bndbox')
            bndbox.appendChild(doc.createElement('xmin')) \
                .appendChild(doc.createTextNode(str(int(box[0]))))
            bndbox.appendChild(doc.createElement('ymin')) \
                .appendChild(doc.createTextNode(str(int(box[1]))))
            bndbox.appendChild(doc.createElement('xmax')) \
                .appendChild(doc.createTextNode(str(int(box[2]))))
            bndbox.appendChild(doc.createElement('ymax')) \
                .appendChild(doc.createTextNode(str(int(box[3]))))
            obj.appendChild(bndbox)

            seg = doc.createElement('segmentation')

            points1 = doc.createElement('points')
            points1.appendChild(doc.createElement('x')) \
                .appendChild(doc.createTextNode(str(int(box[0]))))
            points1.appendChild(doc.createElement('y')) \
                .appendChild(doc.createTextNode(str(int(box[1]))))
            seg.appendChild(points1)

            points2 = doc.createElement('points')
            points2.appendChild(doc.createElement('x')) \
                .appendChild(doc.createTextNode(str(int(box[2]))))
            points2.appendChild(doc.createElement('y')) \
                .appendChild(doc.createTextNode(str(int(box[1]))))
            seg.appendChild(points2)

            points3 = doc.createElement('points')
            points3.appendChild(doc.createElement('x')) \
                .appendChild(doc.createTextNode(str(int(box[2]))))
            points3.appendChild(doc.createElement('y')) \
                .appendChild(doc.createTextNode(str(int(box[3]))))
            seg.appendChild(points3)

            points4 = doc.createElement('points')
            points4.appendChild(doc.createElement('x')) \
                .appendChild(doc.createTextNode(str(int(box[0]))))
            points4.appendChild(doc.createElement('y')) \
                .appendChild(doc.createTextNode(str(int(box[3]))))
            seg.appendChild(points4)
            obj.appendChild(seg)

    return doc


def generate_label_file(result, size, output_label_path, file_path, sub_dir="whole_label"):
    xml_content = generate_whole_xml_contents(result, size)
    output_label_xml_path = os.path.join(output_label_path, sub_dir)

    single_xml_output = os.path.splitext(file_path)[0] + ".xml"
    single_xml_output = os.path.join(output_label_xml_path, single_xml_output)

    if not os.path.isdir(os.path.dirname(single_xml_output)):
        os.makedirs(os.path.dirname(single_xml_output))

    with open(single_xml_output, 'w') as f:
        f.write(xml_content.toprettyxml(indent='        '))


# result format:
# {label_name:[[x1,y1,x2,y2, conf]]}
# regions format: numpy,  [x1, y1, x2, y2]
def get_cropped_region(result, regions):
    regions_box = dict()

    for index in range(regions.shape[0]):

        x1 = regions[index, 0]
        y1 = regions[index, 1]
        x2 = regions[index, 2]
        y2 = regions[index, 3]

        for label_name in result:
            for box in result[label_name]:
                if box[0] >= x1 and box[1] >= y1 and box[2] <= x2 and box[3] <= y2:
                    if index not in regions_box:
                        regions_box[index] = [[box[0], box[1], box[2], box[3], label_name, box[4]]]
                    else:
                        regions_box[index].append([box[0], box[1], box[2], box[3], label_name, box[4]])

    return regions_box


def generate_xml_contents(crop_region_dict, regions):
    xml_content_list = []
    regions_index_list = []

    for region_index in crop_region_dict:
        x1 = regions[region_index, 0]
        y1 = regions[region_index, 1]
        x2 = regions[region_index, 2]
        y2 = regions[region_index, 3]

        boxes_list = crop_region_dict[region_index]

        # xml create
        doc = Document()
        anno = doc.createElement('annotation')
        doc.appendChild(anno)
        filename = doc.createElement('filename')
        filename.appendChild(doc.createTextNode(str(region_index - 1).zfill(8)))
        anno.appendChild(filename)
        relative = doc.createElement('relative')
        relative.appendChild(doc.createElement('x')). \
            appendChild(doc.createTextNode(str(int(x1))))
        relative.appendChild(doc.createElement('y')). \
            appendChild(doc.createTextNode(str(int(y1))))
        relative.appendChild(doc.createElement('rawpath')). \
            appendChild(doc.createTextNode("cropped"))
        anno.appendChild(relative)

        size = doc.createElement('size')
        size.appendChild(doc.createElement('width')).appendChild(doc.createTextNode(str(int(x2 - x1))))
        size.appendChild(doc.createElement('height')).appendChild(doc.createTextNode(str(int(y2 - y1))))

        size.appendChild(doc.createElement('depth')).appendChild(doc.createTextNode(str(3)))
        anno.appendChild(size)

        for box_content in boxes_list:
            obj = doc.createElement('object')
            anno.appendChild(obj)
            obj.appendChild(doc.createElement('name')).appendChild(doc.createTextNode(box_content[4]))
            diff = doc.createElement('difficult');
            diff.appendChild(doc.createTextNode(str(0)))
            trun = doc.createElement('truncated');
            trun.appendChild(doc.createTextNode(str(0)))
            obj.appendChild(diff)
            obj.appendChild(trun)
            bndbox = doc.createElement('bndbox')
            bndbox.appendChild(doc.createElement('xmin')) \
                .appendChild(doc.createTextNode(str(int(box_content[0] - x1))))
            bndbox.appendChild(doc.createElement('ymin')) \
                .appendChild(doc.createTextNode(str(int(box_content[1] - y1))))
            bndbox.appendChild(doc.createElement('xmax')) \
                .appendChild(doc.createTextNode(str(int(box_content[2] - x1))))
            bndbox.appendChild(doc.createElement('ymax')) \
                .appendChild(doc.createTextNode(str(int(box_content[3] - y1))))
            obj.appendChild(bndbox)

        xml_content_list.append(doc)
        regions_index_list.append(region_index)

    return xml_content_list, regions_index_list


def numpy2vips(a):
    height, width, bands = a.shape
    linear = a.reshape(width * height * bands)
    vi = pyvips.Image.new_from_memory(linear.data, width, height, bands,
                                      dtype_to_format[str(a.dtype)])
    return vi


# vips image to numpy array
def vips2numpy(vi):
    return np.ndarray(buffer=vi.write_to_memory(),
                      dtype=format_to_dtype[vi.format],
                      shape=[vi.height, vi.width, vi.bands])


def draw_vips_image(ori_img, all_boxes, rect_color=(128.0, 128.0, 255.0), font_color=(255, 128, 128)):
    img = ori_img.copy()
    font = cv2.FONT_HERSHEY_DUPLEX
    out_rect_color = []
    out_font_color = []
    if 4 == img.bands:
        out_rect_color.extend(rect_color)
        out_rect_color.append(255)
        out_font_color.extend(font_color)
        out_font_color.append(255)
    if 3 == img.bands:
        out_rect_color.extend(rect_color)
        out_font_color.extend(font_color)

    for k in all_boxes:
        for rect in all_boxes[k]:
            possibility = float(rect[-1])
            det_x1 = int(rect[0])
            det_y1 = int(rect[1])
            det_x2 = int(rect[2])
            det_y2 = int(rect[3])

            img = img.draw_rect(out_rect_color, det_x1, det_y1, det_x2 - det_x1, det_y2 - det_y1, fill=False)
            label_np = np.zeros((20, 80, img.bands), dtype=np.uint8)

            cv2.putText(label_np, "{}:{:1.4f}".format(k, float(possibility)), (0, 12), font, 0.33, out_font_color)
            label_image = numpy2vips(label_np);

            img = img.insert(label_image, det_x1, det_y1 - 20, expand=True)

    return img


def init_dict_box(classes):
    new_dict = dict()
    for cls in classes:
        new_dict[cls] = []
    return new_dict


def read_xml(xml_path, CLASSES_NAME):
    et = xml.etree.ElementTree.parse(xml_path)

    obj_dict = init_dict_box(CLASSES_NAME[1:])
    for et_object in et.iter(tag='object'):
        obj_name = et_object.find("name").text
        tmp_bbox = []
        tmp_bbox.append(int(et_object.find("bndbox").find("xmin").text))
        tmp_bbox.append(int(et_object.find("bndbox").find("ymin").text))
        tmp_bbox.append(int(et_object.find("bndbox").find("xmax").text))
        tmp_bbox.append(int(et_object.find("bndbox").find("ymax").text))
        if obj_name in obj_dict:
            obj_dict[obj_name].append(tmp_bbox)

    return obj_dict


def count_obj(result):
    total = 0
    for k in result:
        total += len(result[k])
    return total


def count_unhealth_obj(result, unhealth_list):
    total = 0
    for item in unhealth_list:
        if item in result:
            total += len(result[item])

    return total


def correct_result(tmp_result, bias_value):
    for k in tmp_result:
        for i in range(len(tmp_result[k])):
            tmp_result[k][i][:4] = [x + y for x, y in zip(tmp_result[k][i][:4], bias_value)]
    return tmp_result


def update_result(result, tmp_result):
    for k in result:
        result[k].extend(tmp_result[k])
    return result


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

    regions = np.array(regions, dtype=np.float32)
    return regions


def dump_result_file(result, json_file, unhealth_list):
    dict = {'unhealth': 0, 'result': result};

    count_number = count_unhealth_obj(result, unhealth_list)
    dict['unhealth'] = count_number;

    import json
    with open(json_file, 'w') as f:
        json.dump(dict, f)


def count_result(result):
    dict_count = {};
    for item_key in result:
        dict_count[item_key] = len(result[item_key])
    return dict_count


def cal_iou(box1, box2):
    overlap_x1 = max(box1[0], box2[0])
    overlap_y1 = max(box1[1], box2[1])
    overlap_x2 = min(box1[2], box2[2])
    overlap_y2 = min(box1[3], box2[3])

    tmp_width = overlap_x2 - overlap_x1
    tmp_height = overlap_y2 - overlap_y1

    overlap_width = tmp_width if tmp_width > 0 else 0
    overlap_height = tmp_height if tmp_height > 0 else 0

    overlap_area = overlap_width * overlap_height

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = overlap_area / float((area1 + area2 - overlap_area))

    return iou


def min_iou(box1, box2):
    overlap_x1 = max(box1[0], box2[0])
    overlap_y1 = max(box1[1], box2[1])
    overlap_x2 = min(box1[2], box2[2])
    overlap_y2 = min(box1[3], box2[3])

    tmp_width = overlap_x2 - overlap_x1
    tmp_height = overlap_y2 - overlap_y1

    overlap_width = tmp_width if tmp_width > 0 else 0
    overlap_height = tmp_height if tmp_height > 0 else 0

    overlap_area = overlap_width * overlap_height

    min_area = min(((box2[2] - box2[0] + 1.) * (box2[3] - box2[1] + 1.)),
                   ((box1[2] - box1[0] + 1.) * (box1[3] - box1[1] + 1.)))
    min_overlay = overlap_area / min_area

    return min_overlay


def find_diff_box(base_box_list, comp_box_list, iou_thresh, CLASSES_NAME, function_F):
    diff_box_list = init_dict_box(CLASSES_NAME[0:])

    find_diff = False
    same_boxes = init_dict_box(CLASSES_NAME[0:])
    for obj_name in base_box_list:

        comp_index = []
        for base_box in base_box_list[obj_name]:

            find_box = False

            tmp_index = 0
            for comp_box in comp_box_list[obj_name]:
                iou = function_F(base_box, comp_box)
                if iou >= iou_thresh and tmp_index not in comp_index:
                    find_box = True
                    comp_index.append(tmp_index)
                    same_boxes[obj_name].append(base_box)
                    break
                tmp_index += 1

            if not find_box:
                find_diff = True
                diff_box_list[obj_name].append(base_box)

    return find_diff, diff_box_list, same_boxes


def compare_result(result, gt, CLASSES_NAME, function_F, iou_thresh=0.5, ):
    diff_r1, gt_only_obj_box, same_boxes = find_diff_box(gt, result, iou_thresh, CLASSES_NAME, function_F)
    diff_r2, model_only_obj_box, _ = find_diff_box(result, gt, iou_thresh, CLASSES_NAME, function_F)

    find_diff = diff_r1 or diff_r2
    return find_diff, gt_only_obj_box, model_only_obj_box, same_boxes


def picture_diff_dict(dict_infered_result, dict_gt_result):
    dict_infered = count_result(dict_infered_result)
    dict_gt = count_result(dict_gt_result)

    return_dict = {}
    for item in dict_infered:
        if (item in dict_gt):
            if (dict_infered[item] != 0) and (dict_gt[item] != 0):
                return_dict[item] = "Correct"
            elif (dict_infered[item] != 0) and (dict_gt[item] == 0):
                return_dict[item] = "Incorrect"
            elif (dict_infered[item] == 0) and (dict_gt[item] == 0):
                return_dict[item] = "Correct"
            else:
                return_dict[item] = "Missed"
        else:
            return_dict[item] = "Unlabeled"
    return return_dict


def infer_boxes(box_dict, regions, pyvips_image, det_system):
    for k in range(regions.shape[0]):
        x1, y1, x2, y2 = int(regions[k, 0]), int(regions[k, 1]), \
                         int(regions[k, 2]), int(regions[k, 3])

        pyvips_patch = pyvips_image.extract_area(x1, y1, x2 - x1, y2 - y1)
        patch = vips2numpy(pyvips_patch)
        patch = patch[:, :, ::-1]

        tmp_result = det_system.infer_image(patch)
        tmp_result = correct_result(tmp_result, [x1, y1, x1, y1])

        box_dict = update_result(box_dict, tmp_result)

    return box_dict


def generate_label_files(box_dict, regions, output_label_path, file_path, pyvips_image, sub_img_dir="img",
                         sub_label_dir="label"):
    crop_region_dict = get_cropped_region(box_dict, regions)

    xml_contents, regions_index = generate_xml_contents(crop_region_dict, regions)

    output_label_img_path = os.path.join(output_label_path, sub_img_dir)
    output_label_xml_path = os.path.join(output_label_path, sub_label_dir)

    # write files
    crop_img_counter = 0
    for xml_doc, reg_index in zip(xml_contents, regions_index):

        single_img_output = os.path.splitext(file_path)[0] + "_cropped_{:04d}".format(crop_img_counter) + ".png"
        single_xml_output = os.path.splitext(file_path)[0] + "_cropped_{:04d}".format(crop_img_counter) + ".xml"

        single_img_output = os.path.join(output_label_img_path, single_img_output)
        single_xml_output = os.path.join(output_label_xml_path, single_xml_output)

        if not os.path.isdir(os.path.dirname(single_img_output)):
            os.makedirs(os.path.dirname(single_img_output))

        if not os.path.isdir(os.path.dirname(single_xml_output)):
            os.makedirs(os.path.dirname(single_xml_output))

        # image part
        tmp_pyvips_img = pyvips_image.extract_area(regions[reg_index, 0], regions[reg_index, 1], \
                                                   regions[reg_index, 2] - regions[reg_index, 0], \
                                                   regions[reg_index, 3] - regions[reg_index, 1])
        tmp_pyvips_img.pngsave(single_img_output, Q=100)

        # xml part
        with open(single_xml_output, 'w') as f:
            f.write(xml_doc.toprettyxml(indent='        '))
        crop_img_counter += 1


def main():
    args = parse_args()
    start_time = time.time()

    ext_list = args.exts.split(',')
    ext_list = [ext.lower() for ext in ext_list]
    file_list = scan_files(args.image_path, ext_list)

    if args.iou_cal_method == 'iou':
        function_F = cal_iou
    elif args.iou_cal_method == 'min':
        function_F = min_iou
    else:
        raise Exception('selected iou method not supported.')

    # det_system = alg_system(args)

    if args.count_num:
        count_box = 0
        count_img = 0

    total_count = {}

    unhealth_list = args.unhealth_list

    # 进程数等于gpu数
    NUMBER_OF_PROCESSES = len(args.gpu)
    output_queue = Queue()
    input_queue_list = []
    for i in range(len(args.gpu)):
        input_queue = Queue()
        input_queue_list.append(input_queue)

    for i in range(NUMBER_OF_PROCESSES):
        Process(target=infer_obj_worker, args=(input_queue_list[i], output_queue)).start()

    print("cost time before process images: {}s".format(time.time() - start_time))

    whole_start_time = time.time()

    for file_path in file_list:
        start_time = time.time()
        img_path = os.path.join(args.image_path, file_path)
        pyvips_image = pyvips.Image.new_from_file(img_path)
        print("cost time for opening image: {}s".format(time.time() - start_time))
        print("process {}".format(img_path))

        tmp_start_time = time.time()
        regions = get_regions((pyvips_image.height, pyvips_image.width), args.crop_size, args.overlap)
        print("cost time for getting regions: {}s".format(time.time() - tmp_start_time))

        tmp_start_time = time.time()

        result = init_dict_box(args.CLASSES_NAME[1:])

        # cache
        cache_dir = args.cache_path
        if not os.path.isdir(cache_dir):
            os.makedirs(cache_dir)
        current_pickle_file = os.path.join(cache_dir, os.path.splitext(file_path)[0] + ".pkl")

        if os.path.exists(current_pickle_file):
            print("find pickle file, use it")
            with open(current_pickle_file, 'rb') as f:
                result = pickle.load(f)
        else:
            print("cannot find pickle file, start to infer")
            print('共{}个计算区域（region）'.format(len(regions)))
            start = time.time()
            gpu_list = args.gpu
            for index, single_gpu in enumerate(gpu_list):
                if (index + 1) == len(gpu_list):
                    region_part = regions[int((len(regions) / len(gpu_list)) * index):]
                    print('最后一批regions  ' + str(len(region_part)))
                else:
                    region_part = regions[
                                  (int(len(regions) / len(gpu_list)) * index): int(
                                      (len(regions) / len(gpu_list)) * (index + 1))]
                    print('第{}批regions  '.format(index + 1) + str(len(region_part)))

                for region in region_part:
                    x1, y1, x2, y2 = int(region[0]), int(region[1]), int(region[2]), int(region[3])
                    pyvips_patch = pyvips_image.extract_area(x1, y1, x2 - x1, y2 - y1)
                    patch = vips2numpy(pyvips_patch)

                    gpu_index = gpu_list[index]
                    input_queue_list[index].put([gpu_index, region, patch, args], block=False)

            for i in range(len(regions)):
                tmp_result = output_queue.get()
                result = update_result(result, tmp_result)

            with open(current_pickle_file, 'wb') as f:
                pickle.dump(result, f)

            print('执行时间' + str(time.time() - start))

        result = filter_boxes_dict(result, args.cls_thresh)
        result = nms_in_class(result)
        result = nms_between_classes(result)

        print("cost time for forwarding image: {}s".format(time.time() - tmp_start_time))

        if args.output_label_path is not None and args.label_for_training:
            print("start to generate training label")
            generate_label_files(result, regions, args.output_label_path, file_path, pyvips_image)
            print("generate training label done")

        if args.output_label_path is not None and args.label_for_modifying:
            print("start to generate whole xml file")
            generate_label_file(result, (pyvips_image.width, pyvips_image.height), args.output_label_path, file_path)
            print("generate whole xml file done")

    #     if args.count_num:
    #         # added_num = count_obj(result)
    #         added_num = count_unhealth_obj(result, unhealth_list)
    #         if added_num != 0:
    #             count_img += 1
    #         count_box += added_num
    #
    #     before_draw_cost_time = time.time() - tmp_start_time
    #     print("total cost time without drawing image:", before_draw_cost_time)
    #
    #     single_count_dict = count_result(result)
    #     single_count_dict['total'] = 0
    #
    #     for item in unhealth_list:
    #         if item in single_count_dict:
    #             single_count_dict['total'] += single_count_dict[item]
    #
    #     print(file_path, " file count:", single_count_dict)
    #     for item in single_count_dict:
    #         if item not in total_count:
    #             total_count[item] = single_count_dict[item]
    #         else:
    #             total_count[item] += single_count_dict[item]
    #
    #     if args.xml_path is not None:
    #         gt_result = read_xml(os.path.join(args.xml_path, os.path.splitext(file_path)[0] + ".xml"),
    #                              args.CLASSES_NAME)
    #         gt_count_result = count_result(gt_result)
    #         find_diff, gt_only_obj_box, model_only_obj_box, same_boxes = compare_result(result, gt_result,
    #                                                                                     args.CLASSES_NAME, function_F,
    #                                                                                     args.iou_thresh)
    #         gt_only_obj = count_result(gt_only_obj_box)
    #         model_only_obj = count_result(model_only_obj_box)
    #         same_boxes_result_count = count_result(same_boxes)
    #
    #         if find_diff:
    #             del gt_only_obj['__background__']
    #             del model_only_obj['__background__']
    #             print(file_path, " Missed: ", gt_only_obj)
    #             print(file_path, " Incorrect: ", model_only_obj)
    #
    #         print(file_path, " Ground_truth: ", gt_count_result)
    #         print(file_path, " Same boxes: ", same_boxes_result_count)
    #
    #         picture_result_dict = picture_diff_dict(result, gt_result)
    #         print(file_path, " picture: ", picture_result_dict)
    #
    #     if args.draw_image:
    #         output_path = os.path.join(args.output_path, os.path.splitext(file_path)[0] + ".png")
    #         if not os.path.isdir(os.path.dirname(output_path)):
    #             os.makedirs(os.path.dirname(output_path))
    #
    #         if args.xml_path is not None:
    #             pyvips_image = draw_vips_image(pyvips_image, gt_only_obj_box, (255, 64, 64))
    #             pyvips_image = draw_vips_image(pyvips_image, model_only_obj_box, (64, 255, 64))
    #         else:
    #             pyvips_image = draw_vips_image(pyvips_image, result)
    #
    #         if args.divide <= 1:
    #             pyvips_image.pngsave(output_path, Q=50)
    #         else:
    #             width = pyvips_image.width
    #             height = pyvips_image.height
    #
    #             single_width = int(width / args.divide) + 1
    #             single_height = int(height / args.divide) + 1
    #
    #             for n in range(args.divide):
    #                 for m in range(args.divide):
    #                     tmp_x2 = min(max(0, m * single_width + single_width), width)
    #                     tmp_y2 = min(max(0, n * single_height + single_height), height)
    #                     tmp_x1 = min(max(0, tmp_x2 - single_width), width)
    #                     tmp_y1 = min(max(0, tmp_y2 - single_height), height)
    #
    #                     tmp_pyvips_img = pyvips_image.extract_area(tmp_x1, tmp_y1, tmp_x2 - tmp_x1, tmp_y2 - tmp_y1)
    #
    #                     single_output = os.path.splitext(output_path)[0] + "_cropped_{:04d}".format(
    #                         int(n * args.divide + m)) + os.path.splitext(output_path)[1]
    #
    #                     tmp_pyvips_img.pngsave(single_output, Q=100)
    #
    # cost_time = time.time() - whole_start_time
    #
    # if args.count_num:
    #     total_count['total'] = 0
    #     for item in unhealth_list:
    #         if item in total_count:
    #             total_count['total'] += total_count[item]
    #     print('total count:', total_count)
    #     # print("total number of boxes is {}".format(str(count_box)))
    #     # print("total number of images should be checked is {}".format(str(count_img)))
    #     # print("total number of images is {}".format(str(len(file_list))))
    # print("cost time is {}".format(str(cost_time)))
    # print("avg time is {}".format(str(cost_time / len(file_list))))
    for i in range(NUMBER_OF_PROCESSES):
        input_queue_list[i].put('STOP')


if __name__ == '__main__':
    main()
