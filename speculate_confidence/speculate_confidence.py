# -*- coding: utf-8 -*-
import argparse
import os
import pickle
import time
import xml.etree.cElementTree as ET
from functools import partial
from multiprocessing import cpu_count
from multiprocessing.pool import Pool

import matplotlib.pyplot as plt
import numpy as np
import yaml


def parse_xml(xml_path, target_label):
    root = ET.parse(xml_path)
    objects = root.findall('object')
    annotation_list = []
    for obj in objects:
        label = obj.find('name').text
        if label == target_label:
            bbox_doc = obj.find('bndbox')
            x1 = float(bbox_doc.find('xmin').text)
            y1 = float(bbox_doc.find('ymin').text)
            x2 = float(bbox_doc.find('xmax').text)
            y2 = float(bbox_doc.find('ymax').text)
            bbox = [x1, y1, x2, y2]
            annotation_list.append(bbox)
    return annotation_list


def get_all_xml_regions_of_target_label(xml_file_directory, target_label):
    file_list = os.listdir(xml_file_directory)
    xml_file_list = []
    for file in file_list:
        if file.split('.')[-1].lower() == 'xml':
            xml_file_list.append(file)
    dict = {}
    for xml_file in xml_file_list:
        xml_path = os.path.join(xml_file_directory, xml_file)
        regions = parse_xml(xml_path, target_label)
        file_name = xml_file.split('.')[0]
        dict[file_name] = regions
    return dict


def calc_pre_recall_F1_by_fixed_conf(confidence, pickle_file_directory, pkl_file_list, doctor_regions_dict,
                                     target_label):
    Total_TP1, Total_TP2, Total_FP, Total_FN = 0, 0, 0, 0
    for file in pkl_file_list:
        file_name = file.split('.')[0]
        try:
            doctor_region_list = doctor_regions_dict[file_name]
        except KeyError:
            msg = 'No {} doctor xml'.format(file_name)
            raise Exception(msg)
        with open(os.path.join(pickle_file_directory, file), 'rb') as f:
            result = pickle.load(f)
            computer_region_list = result[target_label]
            TP1, TP2, FP, FN = handle_result(computer_region_list, doctor_region_list, confidence)
        Total_TP1 += TP1
        Total_TP2 += TP2
        Total_FP += FP
        Total_FN += FN
    precision = calc_precision(Total_TP1, Total_FP)
    recall = calc_recall(Total_TP2, Total_FN)
    F1 = calc_F1(precision, recall)
    return precision, recall, F1


def get_precisions_recalls_F1s_by_confidences(sorted_confidence_list, pickle_file_directory, pkl_file_list,
                                              doctor_regions_dict, target_label):
    _calc_pre_recall_F1_by_fixed_conf = partial(calc_pre_recall_F1_by_fixed_conf,
                                                pickle_file_directory=pickle_file_directory,
                                                pkl_file_list=pkl_file_list,
                                                doctor_regions_dict=doctor_regions_dict, target_label=target_label)
    pool = Pool(cpu_count())
    precision_recall_F1_list = pool.map(_calc_pre_recall_F1_by_fixed_conf, sorted_confidence_list)
    pool.close()
    pool.join()

    precision_list, recall_list, F1_list = [], [], []
    for i in precision_recall_F1_list:
        precision_list.append(i[0])
        precision_list.append(i[1])
        precision_list.append(i[2])
    return precision_list, recall_list, F1_list


def for_each_pickle_file(pickle_file_directory, xml_file_directory, target_label):
    pkl_file_list = get_pickle_file_list(pickle_file_directory)
    # 获取所有image的置信度列表
    sorted_confidence_list = get_all_image_region_confidence(pickle_file_directory, target_label)
    ###############################################################################
    # 截取部分confidence在0.6以上的部分
    sorted_confidence_list = np.array(sorted_confidence_list)
    sorted_confidence_list = sorted_confidence_list[sorted_confidence_list >= 0.9]
    print('confidence_length', len(sorted_confidence_list))
    ###############################################################################

    # 先将所有的doctor_xml文件中target_label对应的regions整理成字典{'file1': [], 'file2': [], ...}
    all_doctor_xml_regions_for_single_label = get_all_xml_regions_of_target_label(xml_file_directory, target_label)

    print('开始计算')
    start_time = time.time()
    precision_list, recall_list, F1_list = get_precisions_recalls_F1s_by_confidences(sorted_confidence_list,
                                                                                     pickle_file_directory,
                                                                                     pkl_file_list,
                                                                                     all_doctor_xml_regions_for_single_label,
                                                                                     target_label)
    print('用时{}s'.format(time.time() - start_time))
    result_dict = {}
    result_dict['label'] = target_label
    result_dict['confidence_list'] = sorted_confidence_list
    result_dict['precision_list'] = precision_list
    result_dict['recall_list'] = recall_list
    result_dict['F1_list'] = F1_list
    return result_dict


def get_all_image_region_confidence(pickle_file_directory, label):
    pkl_file_list = get_pickle_file_list(pickle_file_directory)
    all_computer_region_list = []
    for file in pkl_file_list:
        pickle_path = os.path.join(pickle_file_directory, file)
        with open(pickle_path, 'rb') as f:
            result = pickle.load(f)
            computer_region_list = result[label]
            all_computer_region_list += computer_region_list
    all_confidence = [i[-1] for i in all_computer_region_list]
    all_confidence = sorted(all_confidence)
    return all_confidence


def show_result(result_list):
    for result in result_list:
        label = result['label']
        color = result['color']
        confidence_list = result['confidence_list']
        precision_list = result['precision_list']
        recall_list = result['recall_list']
        F1_list = result['F1_list']
        plt.plot(confidence_list, F1_list, color=color, linestyle='solid', label='F1')
        plt.plot(confidence_list, precision_list, color=color, linestyle='dashed', label='precision')
        plt.plot(confidence_list, recall_list, color=color, linestyle='dotted', label='recall')
        plt.xlabel('confidence')
        plt.ylabel(label)
        plt.show()


def get_pickle_file_list(pickle_directory):
    pickle_file_list = []
    for root, dirs, files in os.walk(pickle_directory):
        for file in files:
            if file.split('.')[-1].lower() == 'pkl':
                pickle_file_list.append(file)
    return pickle_file_list


def get_coincide_region_num(current_region_list, doctor_region_list):
    gt_bbox = np.array(doctor_region_list)
    mc_bbox = np.array(current_region_list).reshape((-1, 1, 5))[:, :, :4]
    xmin = np.maximum(gt_bbox[:, 0], mc_bbox[:, :, 0])
    ymin = np.maximum(gt_bbox[:, 1], mc_bbox[:, :, 1])
    xmax = np.minimum(gt_bbox[:, 2], mc_bbox[:, :, 2])
    ymax = np.minimum(gt_bbox[:, 3], mc_bbox[:, :, 3])
    w = np.maximum(xmax - xmin, 0.)
    h = np.maximum(ymax - ymin, 0.)
    inter = w * h
    inter_check = np.where(inter > 0, 1, 0)
    tp_array_1 = np.sum(inter_check, axis=1)
    tp_array_1 = np.where(tp_array_1 > 0, 1, 0)
    TP1 = np.sum(tp_array_1)

    TP2 = np.sum(inter_check, axis=0)
    TP2 = np.where(TP2 > 0, 1, 0)
    TP2 = np.sum(TP2)
    FP = len(current_region_list) - TP1
    FN = len(doctor_region_list) - TP2
    return TP1, TP2, FP, FN


def handle_result(computer_region_list, doctor_region_list, init_confidence):
    # 根据置信度排序
    sorted_all_regions = sorted(computer_region_list, key=lambda x: x[-1])
    current_region_list = []
    for index, region in enumerate(sorted_all_regions):
        if region[-1] >= init_confidence:
            current_region_list = sorted_all_regions[index:]
            break
    TP1, TP2, FP, FN = get_coincide_region_num(current_region_list, doctor_region_list)
    return TP1, TP2, FP, FN


def calc_precision(TP, FP):
    return float('%.4f' % (TP / (TP + FP)))


def calc_recall(TP, FN):
    return float('%.4f' % (TP / (TP + FN)))


def calc_F1(P, R):
    try:
        F1 = float('%.4f' % (2 * P * R / (P + R)))
        return F1
    except ZeroDivisionError:
        return 0


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('yml_path', type=str, help='path to pkl_files')
    args = parser.parse_args()
    yml_file = open(args.yml_path, encoding='utf-8')
    param_dict = yaml.safe_load(yml_file)
    for item in param_dict:
        parser.add_argument(item, type=type(param_dict[item]), default=param_dict[item])
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arg()
    pickle_file_directory = args.pkl_file_directory
    xml_file_directory = args.xml_file_directory
    label_list = args.label_list
    label_color_dict = args.label_color
    # 根据label分类
    result_list = []
    for label in label_list:
        result = for_each_pickle_file(pickle_file_directory, xml_file_directory, label)
        label_color = label_color_dict[label]
        result['color'] = label_color
        result_list.append(result)
    # 展现结果
    show_result(result_list)
