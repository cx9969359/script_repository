# -*- coding: utf-8 -*-
import argparse
import datetime
import os
import pickle
import random
import time
import xml.etree.cElementTree as ET
from functools import partial
from multiprocessing import cpu_count
from multiprocessing.pool import Pool

import matplotlib.pyplot as plt
import numpy as np
import yaml

OFFSET = 1e-8


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
        recall_list.append(i[1])
        F1_list.append(i[2])
    return precision_list, recall_list, F1_list


def for_each_pickle_file(pickle_file_directory, xml_file_directory, target_label, confidence_offset):
    pkl_file_list = get_pickle_file_list(pickle_file_directory)
    # 获取所有image的置信度列表
    sorted_confidence_list = get_all_image_region_confidence(pickle_file_directory, target_label)
    ###############################################################################
    # 截取部分confidence在0.6以上的部分
    sorted_confidence_list = np.array(sorted_confidence_list)
    sorted_confidence_list = sorted_confidence_list[sorted_confidence_list >= confidence_offset]
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


def show_result(show_image, result_list, label_group, output_image_path):
    # 每一个label_group一个图形
    for group in label_group:
        plt.figure(figsize=(12, 6))
        plt.xlabel('confidence')
        plt.yticks(np.arange(0, 1, 0.05))

        for label in group:
            result = {}
            for i in result_list:
                if i['label'] == label:
                    result = i
                    break
            if not result:
                continue
            f1_label = result['label'] + ': F1'
            precision_label = result['label'] + ': precision'
            recall_label = result['label'] + ': recall'
            plt.plot(result['confidence_list'], result['precision_list'], color=result['color'], linestyle='solid',
                     label=precision_label)
            plt.plot(result['confidence_list'], result['recall_list'], color=result['color'], linestyle='dotted',
                     label=recall_label)
            plt.plot(result['confidence_list'], result['F1_list'], color=result['color'], linestyle='-.',
                     label=f1_label)
        plt.legend(loc=0)
        if (show_image):
            plt.show()
        else:
            date = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            file_name = '{}-{}.png'.format(date, random.random())
            save_path = os.path.join(output_image_path, file_name)
            plt.savefig(save_path, dpi=300)


def get_pickle_file_list(pickle_directory):
    pickle_file_list = []
    for root, dirs, files in os.walk(pickle_directory):
        for file in files:
            if file.split('.')[-1].lower() == 'pkl':
                pickle_file_list.append(file)
    return pickle_file_list


def get_coincide_region_num(current_region_list, doctor_region_list):
    if len(doctor_region_list) == 0 and len(current_region_list) != 0:
        TP1, TP2, FN = 0, 0, 0
        FP = len(current_region_list) - 0
        return TP1, TP2, FP, FN
    if (len(doctor_region_list), len(current_region_list)) == (0, 0):
        return 0, 0, 0, 0
    if len(doctor_region_list) != 0 and len(current_region_list) == 0:
        TP1, TP2, FP = 0, 0, 0
        FN = len(doctor_region_list)
        return TP1, TP2, FP, FN

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
    return float('%.4f' % (TP / (TP + FP + OFFSET)))


def calc_recall(TP, FN):
    return float('%.4f' % (TP / (TP + FN + OFFSET)))


def calc_F1(P, R):
    return float('%.4f' % (2 * P * R / (P + R + OFFSET)))


def divide_pkl_label_into_groups(pkl_label_set, group_size):
    """
    将pickle_label分组成新的need_label_group
    :param pkl_label_set:
    :return:
    """
    pkl_label_list = list(pkl_label_set)
    group = []
    for index, value in enumerate(pkl_label_list):
        if (index + 1) * group_size >= len(pkl_label_list):
            part = pkl_label_list[index * group_size:]
            group.append(part)
            break
        else:
            part = pkl_label_list[index * group_size: (index + 1) * group_size]
            group.append(part)
    return group


def trim_label_group(need_label_group, pkl_label_set, group_size):
    """
    如果没有配置label_group, 则使用pickle文件中的label, 并将pickle_label分组
    :param need_label_group:
    :param pkl_label_set:
    :return:
    """
    if not need_label_group:
        label_group = divide_pkl_label_into_groups(pkl_label_set, group_size)
        return pkl_label_set, label_group
    else:
        label_list = []
        for group in need_label_group:
            for label in group:
                label_list.append(label)
        return set(label_list), need_label_group


def get_all_pkl_label(pickle_file_directory, pkl_file_list):
    label_list = []
    for file in pkl_file_list:
        pkl_path = os.path.join(pickle_file_directory, file)
        with open(pkl_path, 'rb') as f:
            result = pickle.load(f)
            for k, v in result.items():
                label_list.append(k)
    return set(label_list)


def save_cache_as_pkl(save_directory, result, label):
    if save_directory:
        date = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        file_name = '{}-{}.pkl'.format(date, label)
        file_path = os.path.join(save_directory, file_name)
        with open(file_path, 'wb') as f:
            pickle.dump(result, f)


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
    confidence_offset = args.confidence_offset
    need_label_group = args.label_group
    label_color_dict = args.label_color

    pkl_file_list = get_pickle_file_list(pickle_file_directory)
    pkl_label_set = get_all_pkl_label(pickle_file_directory, pkl_file_list)
    print(pkl_label_set)

    # 根据label分类
    need_label_set, need_label_group = trim_label_group(need_label_group, pkl_label_set, args.group_size)
    result_list = []
    for label in need_label_set:
        result = for_each_pickle_file(pickle_file_directory, xml_file_directory, label, confidence_offset)
        label_color = label_color_dict[label]
        result['color'] = label_color
        save_cache_as_pkl(args.result_cache_directory, result, label)
        result_list.append(result)
    # 展现结果
    show_image = args.show_image
    output_image_path = args.output_image_path
    show_result(show_image, result_list, need_label_group, output_image_path)
