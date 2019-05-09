from util.service.time_decorator import spend_time


@spend_time
def get_function_by_iou_cal(args):
    if args.iou_cal_method == 'iou':
        function_F = cal_iou
    elif args.iou_cal_method == 'min':
        function_F = min_iou
    else:
        raise Exception('selected iou method not supported.')
    return function_F


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
