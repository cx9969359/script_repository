# -*- coding: utf-8 -*-
import argparse
import os
import pickle
import xml.etree.cElementTree as ET

import matplotlib.pyplot as plt


def get_doctor_regions_by_xml(xml_file_directory, file):
    if not os.path.isdir(xml_file_directory):
        msg = 'xml_directory ({}) is error'.format(xml_file_directory)
        raise Exception(msg)
    xml_file_name = file.split('.')[0] + '.xml'
    xml_files = os.listdir(xml_file_directory)
    if xml_file_name not in xml_files:
        msg = 'No doctor_xml for {}'.format(file)
        raise Exception(msg)
    xml_path = os.path.join(xml_file_directory, xml_file_name)
    doctor_region_list = parse_xml(xml_path)
    return doctor_region_list


def parse_xml(xml_path):
    root = ET.parse(xml_path)
    objects = root.findall(root)
    annotation_list = []
    for obj in objects:
        label = obj.find('name').text
        bbox_doc = obj.find('bndbox')
        x1 = float(bbox_doc.find('xmin').text)
        y1 = float(bbox_doc.find('ymin').text)
        x2 = float(bbox_doc.find('xmax').text)
        y2 = float(bbox_doc.find('ymax').text)
        bbox = [x1, y1, x2, y2]
        annotation_list.append(bbox)
    return annotation_list


def for_each_pickle_file(pickle_file_directory, xml_file_directory):
    pkl_file_list = get_pickle_file_list(pickle_file_directory)
    for file in pkl_file_list:
        doctor_region_list = get_doctor_regions_by_xml(xml_file_directory, file)
        pickle_path = os.path.join(pickle_file_directory, file)
        with open(pickle_path, 'rb') as f:
            result = pickle.load(f)
            computer_region_list = []
            for single_class in result:
                regions = []
                for key, value in single_class.items():
                    regions += value
                computer_region_list += regions
        handled_list = handle_result(computer_region_list, doctor_region_list)
        show_result(handled_list)


def show_result(handled_list):
    confidence_list = [i['confidence'] for i in handled_list]
    F1_list = [i['F1'] for i in handled_list]
    plt.plot(confidence_list, F1_list)
    plt.show()


def get_pickle_file_list(pickle_directory):
    pickle_file_list = []
    for root, dirs, files in os.walk(pickle_directory):
        for file in files:
            if file.split('.')[-1].lower() == 'pkl':
                pickle_file_list.append(file)
    return pickle_file_list


def handle_result(computer_region_list, doctor_region_list):
    # 根据置信度排序
    handled_list = []
    sorted_regions_by_confidence = sorted(computer_region_list, key=lambda x: x[-1])
    for index, region in enumerate(sorted_regions_by_confidence):
        dict = {}
        dict['confidence'] = region[-1]
        current_region_list = sorted_regions_by_confidence[index:]
        correct_num = 0
        for comp_region in current_region_list:
            comp_region = comp_region[:-1]
            for doctor_region in doctor_region_list:
                # 判断是否相交
                crossing = infer_crossing(comp_region, doctor_region)
                if (crossing):
                    overlap = get_overlap_area(comp_region, doctor_region)
                    if overlap >= 0.01:
                        correct_num += 1
                        break
        print('current_correct_num', correct_num)
        TP = correct_num
        FP = len(current_region_list) - correct_num
        FN = len(doctor_region_list) - correct_num
        precision = calc_precision(TP, FP)
        recall = calc_recall(TP, FN)
        harmmean_F1 = calc_F1(precision, recall)
        dict['precision'] = precision
        dict['recall'] = recall
        dict['F1'] = harmmean_F1
        handled_list.append(dict)
    return handled_list


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


def get_overlap_area(region1, region2):
    x01, y01, x02, y02 = region1
    x11, y11, x12, y12 = region2
    col = min(x02, x12) - max(x01, x11)
    row = min(y02, y12) - max(y01, y11)
    intersection = col * row
    area2 = (x12 - x11) * (y12 - y11)
    overlap = intersection / area2
    return overlap


def calc_precision(TP, FP):
    return float('%.4f' % (TP / (TP + FP)))


def calc_recall(TP, FN):
    return float('%.4f' % (TP / (TP + FN)))


def calc_F1(P, R):
    return float('%.4f' % (2 * P * R / (P + R)))


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('pkl_file_directory', type=str, help='path to pkl_files')
    parser.add_argument('xml_file_directory', type=str, help='path to xml_files')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arg()
    pickle_file_directory = args.pkl_file_directory
    xml_file_directory = args.xml_file_directory
    pkl_file_list = get_pickle_file_list(pickle_file_directory)
    # 求单个pickle文件
    for_each_pickle_file(pickle_file_directory, xml_file_directory)
    # 根据敏感度及置信度处理result
