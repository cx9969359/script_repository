import copy

import numpy as np

# box[0]:x1, box[1]:y1, box[2]:x2, box[3]:y2
# point x1,y1 is left top point of box, and point x2,y2 is right
from util.service.time_decorator import spend_time


def cal_iou_small_based(box1, box2):
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

    small_area = min(area1, area2)
    iou = overlap_area / small_area
    return iou


def filter_boxes(all_boxes, num_classes, probility_nms_threshold):
    filtered_boxes = copy.deepcopy(all_boxes)
    # filter all boxes which is lower than threshold
    for x in range(1, num_classes):
        y = 0
        while y < len(filtered_boxes[x]):
            possibility = filtered_boxes[x][y][4]
            if possibility < probility_nms_threshold:
                tmp = filtered_boxes[x].tolist()
                del tmp[y]
                filtered_boxes[x] = np.asarray(tmp)
                # filtered_boxes[x] = np.delete(filtered_boxes[x], y)
                continue
            y += 1
    return filtered_boxes


@spend_time
def filter_boxes_dict(box_result_dict, probility_nms_threshold):
    filtered_boxes = copy.deepcopy(box_result_dict)

    for key in filtered_boxes:
        y = 0
        while y < len(filtered_boxes[key]):
            possibility = filtered_boxes[key][y][4]
            if possibility < probility_nms_threshold:
                tmp = filtered_boxes[key]
                del tmp[y]
                filtered_boxes[key] = tmp
                continue

            y += 1
    return filtered_boxes


def filter_objs_in_class_part(label_key, n, filtered_boxes, nms_threshold):
    obj1 = filtered_boxes[label_key][n]

    # nms in class
    j = 0
    while j < len(filtered_boxes[label_key]):
        if j == n:
            j += 1
            continue

        obj2 = filtered_boxes[label_key][j]
        iou = cal_iou_small_based(obj1, obj2)
        if iou >= nms_threshold:
            if obj1[4] < obj2[4]:
                pass;
            else:
                tmp = filtered_boxes[label_key]
                tmp.remove(tmp[j])
                filtered_boxes[label_key] = tmp
                continue
        j += 1
    return filtered_boxes


def takeLast(elem):
    return elem[-1]


@spend_time
def nms_in_class(all_boxes):
    filtered_boxes = all_boxes

    # nms in class
    for m in filtered_boxes:
        n = 0
        filtered_boxes[m].sort(key=takeLast, reverse=True)

        while n < len(filtered_boxes[m]):
            # if result is true, check next obj, else, delete current obj
            result = filter_objs_in_class_part(m, n, filtered_boxes, 0.5)

            if not result:
                tmp = filtered_boxes[m].tolist()
                del tmp[n]
                filtered_boxes[m] = np.asarray(tmp)
                continue
            n += 1
    return filtered_boxes


@spend_time
def nms_between_classes(filtered_boxes):
    for m in filtered_boxes:
        n = 0
        while n < len(filtered_boxes[m]):
            # if result is true, check next obj, else, delete current obj
            result = filter_objs_between_classes(m, n, filtered_boxes, 0.5)

            if not result:
                tmp = filtered_boxes[m]
                # del tmp[n]
                tmp.remove(tmp[n])
                filtered_boxes[m] = tmp
                continue
            n += 1

    return filtered_boxes


def filter_objs_between_classes(m, n, filtered_boxes, nms_threshold):
    obj1 = filtered_boxes[m][n]

    # nms between class
    for k in filtered_boxes:
        if k == m:
            continue

        j = 0
        while j < len(filtered_boxes[k]):

            obj2 = filtered_boxes[k][j]

            iou = cal_iou_small_based(obj1, obj2)
            if iou >= nms_threshold:
                if obj1[4] < obj2[4]:
                    return False
                else:
                    tmp = filtered_boxes[k]
                    tmp.remove(tmp[j])

                    filtered_boxes[k] = tmp
                    continue

            j += 1

    return True
