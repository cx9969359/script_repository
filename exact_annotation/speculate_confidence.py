# -*- coding: utf-8 -*-
import argparse
import os
import pickle
import xml.etree.cElementTree as ET

import matplotlib.pyplot as plt
import yaml


def get_doctor_regions_by_xml(xml_file_directory, file, target_label):
    if not os.path.isdir(xml_file_directory):
        msg = 'xml_directory ({}) is error'.format(xml_file_directory)
        raise Exception(msg)
    xml_file_name = file.split('.')[0] + '.xml'
    xml_files = os.listdir(xml_file_directory)
    if xml_file_name not in xml_files:
        msg = 'No doctor_xml for {}'.format(file)
        raise Exception(msg)
    xml_path = os.path.join(xml_file_directory, xml_file_name)
    doctor_region_list = parse_xml(xml_path, target_label)
    return doctor_region_list


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


def for_each_pickle_file(pickle_file_directory, xml_file_directory, target_label):
    pkl_file_list = get_pickle_file_list(pickle_file_directory)
    # 获取所有image的置信度列表
    all_confidence = get_all_image_region_confidence(pickle_file_directory, target_label)
    precision_list, recall_list, F1_list = [], [], []
    for confidence in all_confidence:
        Total_TP, Total_FP, Total_FN = 0, 0, 0
        for file in pkl_file_list:
            doctor_region_list = get_doctor_regions_by_xml(xml_file_directory, file, target_label)
            with open(os.path.join(pickle_file_directory, file), 'rb') as f:
                result = pickle.load(f)
                computer_region_list = result[target_label]
                TP, FP, FN = handle_result(computer_region_list, doctor_region_list, confidence)
            Total_TP += TP
            Total_FP += FP
            Total_FN += FN
        precision = calc_precision(Total_TP, Total_FP)
        precision_list.append(precision)
        recall = calc_recall(Total_TP, Total_FN)
        recall_list.append(recall)
        F1 = calc_F1(precision, recall)
        F1_list.append(F1)
    result = {}
    result['label'] = target_label
    result['confidence_list'] = all_confidence
    result['precision_list'] = precision_list
    result['recall_list'] = recall_list
    result['F1_list'] = F1_list
    return result
    # show_result(all_confidence, precision_list, recall_list, F1_list, target_label)


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


def show_result(all_confidence, precision_list, recall_list, F1_list, target_label):
    plt.plot(all_confidence, F1_list, color='g', linestyle='solid', label='F1')
    plt.plot(all_confidence, precision_list, color='r', linestyle='dashed', label='precision')
    plt.plot(all_confidence, recall_list, color='b', linestyle='dotted', label='recall')
    plt.xlabel('confidence')
    plt.ylabel(target_label)
    plt.show()


def get_pickle_file_list(pickle_directory):
    pickle_file_list = []
    for root, dirs, files in os.walk(pickle_directory):
        for file in files:
            if file.split('.')[-1].lower() == 'pkl':
                pickle_file_list.append(file)
    return pickle_file_list


def get_correct_region_num(current_region_list, doctor_region_list):
    correct_num = 0
    for comp_region in current_region_list:
        comp_region = comp_region[:-1]
        for doctor_region in doctor_region_list:
            # 判断是否相交
            crossing = infer_crossing(comp_region, doctor_region)
            if (crossing):
                correct_num += 1
                break
    return correct_num


def handle_result(computer_region_list, doctor_region_list, init_confidence):
    # 根据置信度排序
    sorted_all_regions = sorted(computer_region_list, key=lambda x: x[-1])
    current_region_list = []
    for index, region in enumerate(sorted_all_regions):
        if region[-1] >= init_confidence:
            current_region_list = sorted_all_regions[index:]
            break
    correct_num = get_correct_region_num(current_region_list, doctor_region_list)

    TP = correct_num
    FP = len(current_region_list) - correct_num
    FN = len(doctor_region_list) - correct_num
    return TP, FP, FN


def infer_crossing(region1, region2):
    x01, y01, x02, y02 = region1
    x11, y11, x12, y12 = region2
    lx = abs((x01 + x02) / 2 - (x11 + x12) / 2)
    ly = abs((y01 + y02) / 2 - (y11 + y12) / 2)
    sax = abs(x01 - x02)
    sbx = abs(x11 - x12)
    say = abs(y01 - y02)
    sby = abs(y11 - y12)
    if lx <= (sax + sbx) / 2 and ly <= (say + sby) / 2:
        return True
    else:
        return False


def calc_precision(TP, FP):
    return float('%.4f' % (TP / (TP + FP)))


def calc_recall(TP, FN):
    return float('%.4f' % (TP / (TP + FN)))


def calc_F1(P, R):
    return float('%.4f' % (2 * P * R / (P + R)))


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
    label_color_dict = args.label_color_dict
    # 根据label分类
    result_list = []
    for label in label_list:
        result = for_each_pickle_file(pickle_file_directory, xml_file_directory, label)
        label_color = label_color_dict['label']
        result_list += result
    # 展现结果
    show_result(result_list)
