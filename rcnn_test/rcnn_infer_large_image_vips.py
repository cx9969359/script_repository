import argparse
import ast
import os
import pickle

import pyvips
import yaml

from alg_system import alg_system
from util.nms_filter import nms_in_class, nms_between_classes, filter_boxes_dict
from util.service.generate_label.read_xml import init_dict_box
from util.service.generate_label.tile import generate_label_files
from util.service.generate_label.whole import generate_label_file
from util.service.infer import infer_boxes
from util.service.regions import get_regions
from util.service.scan import scan_files


def parse_args():
    parser = argparse.ArgumentParser(description='Test a Faster R-CNN network',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('yml_path', type=str, help='path to image')

    args = parser.parse_args()
    yml_file = open(args.yml_path, encoding='utf-8')

    param_dict = yaml.safe_load(yml_file)
    for item in param_dict:
        print('key', item, 'type:', type(param_dict[item]), 'value:', param_dict[item])
        parser.add_argument(item, type=type(param_dict[item]), default=param_dict[item], help='')
    args = parser.parse_args()
    parser.add_argument('--rcnn_num_classes', type=int, default=len(args.CLASSES_NAME), help='')
    args = parser.parse_args()
    args.img_pixel_means = ast.literal_eval(args.img_pixel_means)
    args.img_pixel_stds = ast.literal_eval(args.img_pixel_stds)
    args.rpn_anchor_scales = ast.literal_eval(args.rpn_anchor_scales)
    args.rpn_anchor_ratios = ast.literal_eval(args.rpn_anchor_ratios)
    args.rcnn_pooled_size = ast.literal_eval(args.rcnn_pooled_size)
    args.rcnn_bbox_stds = ast.literal_eval(args.rcnn_bbox_stds)
    return args


def main():
    args = parse_args()
    file_list = scan_files(args)

    # 读图、切图
    for file_path in file_list:
        img_path = os.path.join(args.image_path, file_path)
        pyvips_image = pyvips.Image.new_from_file(img_path)
        regions = get_regions((pyvips_image.height, pyvips_image.width), args.crop_size, args.overlap)
        print('共{}个计算区域（region）'.format(len(regions)))

        # cache and train
        cache_dir = args.cache_path
        if not os.path.isdir(cache_dir):
            os.makedirs(cache_dir)
        current_pickle_file = os.path.join(cache_dir, os.path.splitext(file_path)[0] + ".pkl")

        if os.path.exists(current_pickle_file):
            print("Use pickle file")
            with open(current_pickle_file, 'rb') as f:
                result = pickle.load(f)
        else:
            print("No pickle, start to infer")
            result = init_dict_box(args.CLASSES_NAME[1:])
            det_system = alg_system(args, args.gpu[0])
            result = infer_boxes(result, regions, pyvips_image, det_system)
            with open(current_pickle_file, 'wb') as f:
                pickle.dump(result, f)

        # 过滤bbox
        result = filter_boxes_dict(result, args.cls_thresh)
        result = nms_in_class(result)
        result = nms_between_classes(result)

        # 获取大小图坐标并修正
        if args.output_label_path is not None and args.label_for_training:
            generate_label_files(result, regions, args.output_label_path, file_path, pyvips_image)

        if args.output_label_path is not None and args.label_for_modifying:
            generate_label_file(result, (pyvips_image.width, pyvips_image.height), args.output_label_path, file_path)

        # 统计，对比标注
        # statistics_label(args, result, file_path, pyvips_image)

    # if args.count_num:
    #     total_count['total'] = 0
    #     for item in unhealth_list:
    #         if item in total_count:
    #             total_count['total'] += total_count[item]
    #     print('total count:', total_count)
    # print("total number of boxes is {}".format(str(count_box)))
    # print("total number of images should be checked is {}".format(str(count_img)))
    # print("total number of images is {}".format(str(len(file_list))))
    # print("cost time is {}".format(str(cost_time)))
    # print("avg time is {}".format(str(cost_time / len(file_list))))


if __name__ == '__main__':
    main()
