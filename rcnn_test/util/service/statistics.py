import os
import cv2
import pyvips

from util.service.generate_label.read_xml import read_xml
from util.service.iou import get_function_by_iou_cal
from util.service.time_decorator import spend_time
import numpy as np

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


@spend_time
def statistics_label(args, result, file_path, pyvips_image):
    count_img = 0
    count_box = 0
    unhealth_list = args.unhealth_list
    if args.count_num:
        # added_num = count_obj(result)
        added_num = count_unhealth_obj(result, unhealth_list)
        if added_num != 0:
            count_img += 1
        count_box += added_num

    single_count_dict = count_result(result)
    single_count_dict['total'] = 0

    for item in unhealth_list:
        if item in single_count_dict:
            single_count_dict['total'] += single_count_dict[item]

    print(file_path, " file count:", single_count_dict)

    total_count = {}
    for item in single_count_dict:
        if item not in total_count:
            total_count[item] = single_count_dict[item]
        else:
            total_count[item] += single_count_dict[item]

    if args.xml_path is not None:
        gt_result = read_xml(args, file_path)
        gt_count_result = count_result(gt_result)
        function_F = get_function_by_iou_cal(args)
        find_diff, gt_only_obj_box, model_only_obj_box, same_boxes = compare_result(result, gt_result,
                                                                                    args.CLASSES_NAME, function_F,
                                                                                    args.iou_thresh)
        gt_only_obj = count_result(gt_only_obj_box)
        model_only_obj = count_result(model_only_obj_box)
        same_boxes_result_count = count_result(same_boxes)

        if find_diff:
            del gt_only_obj['__background__']
            del model_only_obj['__background__']
            print(file_path, " Missed: ", gt_only_obj)
            print(file_path, " Incorrect: ", model_only_obj)

        picture_result_dict = picture_diff_dict(result, gt_result)
        print(file_path, " Ground_truth: ", gt_count_result)
        print(file_path, " Same boxes: ", same_boxes_result_count)
        print(file_path, " picture: ", picture_result_dict)

    if args.draw_image:
        output_path = os.path.join(args.output_path, os.path.splitext(file_path)[0] + ".png")
        if not os.path.isdir(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))

        if args.xml_path is not None:
            pyvips_image = draw_vips_image(pyvips_image, gt_only_obj_box, (255, 64, 64))
            pyvips_image = draw_vips_image(pyvips_image, model_only_obj_box, (64, 255, 64))
        else:
            pyvips_image = draw_vips_image(pyvips_image, result)

        if args.divide <= 1:
            pyvips_image.pngsave(output_path, Q=50)
        else:
            width = pyvips_image.width
            height = pyvips_image.height

            single_width = int(width / args.divide) + 1
            single_height = int(height / args.divide) + 1

            for n in range(args.divide):
                for m in range(args.divide):
                    tmp_x2 = min(max(0, m * single_width + single_width), width)
                    tmp_y2 = min(max(0, n * single_height + single_height), height)
                    tmp_x1 = min(max(0, tmp_x2 - single_width), width)
                    tmp_y1 = min(max(0, tmp_y2 - single_height), height)

                    tmp_pyvips_img = pyvips_image.extract_area(tmp_x1, tmp_y1, tmp_x2 - tmp_x1, tmp_y2 - tmp_y1)

                    single_output = os.path.splitext(output_path)[0] + "_cropped_{:04d}".format(
                        int(n * args.divide + m)) + os.path.splitext(output_path)[1]

                    tmp_pyvips_img.pngsave(single_output, Q=100)


@spend_time
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


def count_unhealth_obj(result, unhealth_list):
    total = 0
    for item in unhealth_list:
        if item in result:
            total += len(result[item])
    return total


def count_result(result):
    dict_count = {}
    for item_key in result:
        dict_count[item_key] = len(result[item_key])
    return dict_count


@spend_time
def compare_result(result, gt, CLASSES_NAME, function_F, iou_thresh=0.5, ):
    diff_r1, gt_only_obj_box, same_boxes = find_diff_box(gt, result, iou_thresh, CLASSES_NAME, function_F)
    diff_r2, model_only_obj_box, _ = find_diff_box(result, gt, iou_thresh, CLASSES_NAME, function_F)

    find_diff = diff_r1 or diff_r2
    return find_diff, gt_only_obj_box, model_only_obj_box, same_boxes


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
            label_image = numpy2vips(label_np)

            img = img.insert(label_image, det_x1, det_y1 - 20, expand=True)

    return img


def init_dict_box(classes):
    new_dict = dict()
    for cls in classes:
        new_dict[cls] = []
    return new_dict


def numpy2vips(a):
    height, width, bands = a.shape
    linear = a.reshape(width * height * bands)
    vi = pyvips.Image.new_from_memory(linear.data, width, height, bands,
                                      dtype_to_format[str(a.dtype)])
    return vi
