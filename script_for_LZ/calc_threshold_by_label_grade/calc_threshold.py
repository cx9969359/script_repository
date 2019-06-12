import argparse
import os
import pickle

import yaml

confidence = -1


def scan_pickle_sub_folder(pickle_file_directory):
    path = os.path.join(pickle_file_directory)
    if not os.path.isdir(path):
        raise Exception('No such pickle_file_directory')
    return os.listdir(pickle_file_directory)


def scan_pickle_file(directory_path):
    path = os.path.join(directory_path)
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.split('.')[-1].lower() == 'pkl':
                file_list.append(file)
    return file_list


def calc_annotation_num(pickle_file_path, label):
    with open(pickle_file_path, 'rb') as f:
        result = pickle.load(f)
        label_list = result.get(label, [])
        label_list = sorted(label_list, key=lambda x: x[-1])
        init_index = 0
        for index, value in enumerate(label_list):
            if value[-1] > confidence:
                init_index = index
                break
        label_list = label_list[init_index:]
        return len(label_list)


def analyze_threshold(result_list, init_index, highest_sensitivity):
    init_threshold = result_list[init_index]
    morbidity = float('%.2f' % ((len(result_list) - init_index) / len(result_list)))
    if morbidity > highest_sensitivity:
        equal_index_list = [i for i, v in enumerate(result_list) if v == init_threshold]
        if (equal_index_list[-1] + 1) >= len(result_list):
            return result_list[-1]
        else:
            init_index = equal_index_list[-1] + 1
            analyze_threshold(result_list, init_index, highest_sensitivity)
            return result_list[init_index]
    else:
        return result_list[init_index]


def trim_sub_folder_by_label_grade(sub_folder_list, grade_label_list):
    _list = []
    for label in grade_label_list:
        if label in sub_folder_list:
            _list.append(label)
    return _list


def get_senior_label_num_list(sub_folder_path, pickle_files):
    label_num_list = []
    for pickle in pickle_files:
        file_path = os.path.join(sub_folder_path, pickle)
        label = os.path.basename(sub_folder_path)
        single_num = calc_annotation_num(file_path, label)
        label_num_list.append(single_num)
    label_num_list = sorted(label_num_list)
    return label_num_list


def get_single_sub_folder_label_num(pickle_files, sub_folder_path, previous_label_list, current_label):
    total_annotation_num = []
    for pickle in pickle_files:
        file_path = os.path.join(sub_folder_path, pickle)
        # 从高到底优先级标注数量
        single_dict = {}
        for label in previous_label_list:
            single_num = calc_annotation_num(file_path, label)
            single_dict[label] = single_num
        single_num = calc_annotation_num(file_path, current_label)
        single_dict[current_label] = single_num
        total_annotation_num.append(single_dict)
    return total_annotation_num


def get_previous_label_list(threshold_dict):
    previous_label_list = []
    for k, v in threshold_dict.items():
        previous_label_list.append(k)
    return previous_label_list


def get_senior_label_threshold(sub_folder_path, pickle_files, threshold_dict):
    # 获取高优先级的标签列表
    previous_label_list = get_previous_label_list(threshold_dict)

    current_label = os.path.basename(sub_folder_path)
    # 求出所有次级pkl文件夹下所有标签的数量，default = {'highest_label': count1, 'secondary_label': count2, ...}
    total_annotation_num = get_single_sub_folder_label_num(pickle_files, sub_folder_path, previous_label_list,
                                                           current_label)
    before_trim_length = len(total_annotation_num)
    # 去除高优先级标注数量超过其阈值的部分
    need_trim_index = []
    for key, value in threshold_dict.items():
        init_label = key
        init_threshold = value
        index_list = [index for index, value in enumerate(total_annotation_num) if
                      value[init_label] >= init_threshold]
        need_trim_index += index_list
    need_trim_index_set = set(need_trim_index)
    count = 0
    for i in need_trim_index_set:
        total_annotation_num.pop(i - count)
        count += 1
    # 取最多，故排序后取最小为阈值
    if len(total_annotation_num) == 0:
        # 剔除完优先级图片之后剩余为0
        msg = 'arithmetic error!  remove previous {} count is zero'.format(current_label)
        raise Exception(msg)

    # 求除最高优先级标签的敏感性
    sensibility = '%.2f' % (len(total_annotation_num) / before_trim_length)
    print('敏感性：{}={}'.format(current_label, sensibility))
    total_annotation_num = sorted(total_annotation_num, key=lambda x: x[current_label])
    threshold = total_annotation_num[0][current_label]
    return threshold


def calc_threshold_dict(root_directory, trim_sub_folder_list, highest_sensitivity):
    threshold_dict = {}
    for index, sub_folder in enumerate(trim_sub_folder_list):
        sub_folder_path = os.path.join(root_directory, sub_folder)
        pickle_files = scan_pickle_file(sub_folder_path)
        # 优先级最高
        if index == 0:
            label_num_list = get_senior_label_num_list(sub_folder_path, pickle_files)
            init_index = 0
            threshold = analyze_threshold(label_num_list, init_index, highest_sensitivity)
            threshold_dict[sub_folder] = threshold
        else:
            threshold = get_senior_label_threshold(sub_folder_path, pickle_files, threshold_dict)
            threshold_dict[sub_folder] = threshold
    return threshold_dict


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
    root_directory = args.pkl_directory
    grade_label_list = args.grade_label_list
    highest_sensitivity = args.highest_sensitivity
    confidence = args.confidence
    sub_folder_list = scan_pickle_sub_folder(root_directory)
    # 根据优先级整理子文件夹
    trim_sub_folder_list = trim_sub_folder_by_label_grade(sub_folder_list, grade_label_list)
    # 求阈值
    threshold_dict = calc_threshold_dict(root_directory, trim_sub_folder_list, highest_sensitivity)
    print(threshold_dict)
