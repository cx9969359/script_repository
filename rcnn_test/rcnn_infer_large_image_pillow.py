import argparse
import xml.etree.ElementTree
import os,time
import ast
import yaml

import cv2
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = 10000000000
import numpy as np

import mxnet as mxnet
from mxnet.module import Module

from symnet.model import load_param, check_shape
from alg_system import alg_system
#from alg_system import CLASSES_NAME
from util.nms_filter import nms_in_class, nms_between_classes;
import pyvips;


def parse_args():
    parser = argparse.ArgumentParser(description='Test a Faster R-CNN network',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('yml_path', type=str, help='path to image')
   
    args = parser.parse_args()
    yml_file = open(args.yml_path, encoding='utf-8')
    
    param_dict = yaml.safe_load(yml_file)  
    for item in param_dict:
        print('key', item,'type:', type(param_dict[item]), 'value:', param_dict[item])
        parser.add_argument(item, type=type(param_dict[item]),default=param_dict[item],  help='');
    args = parser.parse_args()
    parser.add_argument('--rcnn_num_classes', type=int,default=len(args.CLASSES_NAME),  help='');
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
                file_list.append(os.path.join(root, f).replace(os.path.join(input_path, ""), "", 1 ))
    return file_list
    
def draw_image(ori_img, all_boxes, rect_color=(255, 0, 0), font_color=(255, 255, 255)):
    img = ori_img.copy()
    font = cv2.FONT_HERSHEY_DUPLEX

    for k in all_boxes:
        for rect in all_boxes[k]:
            possibility = float(rect[-1])
            det_x1 = int(rect[0])
            det_y1 = int(rect[1])
            det_x2 = int(rect[2])
            det_y2 = int(rect[3])

            img = cv2.rectangle(img, (det_x1, det_y1), (det_x2, det_y2), rect_color, 2)
            img = cv2.putText(img, "{}:{:1.4f}".format(k, float(possibility)), (det_x1, det_y1+20), font, 0.5, font_color)

    return img

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
        if item in result :
            total +=  len(result[item])

    return total

def correct_result(tmp_result, bias_value):
    for k in tmp_result:
        for i in range(len(tmp_result[k])):
            tmp_result[k][i][:4] = [ x+y for x,y in zip(tmp_result[k][i][:4], bias_value)]
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

    """
        assign the bboxes to every cropped region
    """
    regions = np.array(regions, dtype=np.float32)
    return regions

def dump_result_file(result, json_file, unhealth_list): 
    dict = {'unhealth':0, 'result':result};
    
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
    iou = overlap_area / float((area1+area2 - overlap_area))

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

    min_area =  min(((box2[2] - box2[0] + 1.) * (box2[3] - box2[1] + 1.)), ((box1[2] - box1[0] + 1.) *(box1[3] - box1[1]+ 1.)))							
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


def compare_result(result, gt, CLASSES_NAME,  function_F, iou_thresh = 0.5,):

    diff_r1, gt_only_obj_box, same_boxes= find_diff_box(gt, result, iou_thresh, CLASSES_NAME, function_F)
    diff_r2, model_only_obj_box, _ = find_diff_box(result, gt, iou_thresh, CLASSES_NAME, function_F)

    find_diff = diff_r1 or diff_r2
    return find_diff, gt_only_obj_box, model_only_obj_box, same_boxes

def picture_diff_dict(dict_infered_result,dict_gt_result):
    
    dict_infered= count_result(dict_infered_result)
    dict_gt= count_result(dict_gt_result)
    
    return_dict = {};
    for item in dict_infered:
        if (item in dict_gt):
            if(dict_infered[item] != 0) and (dict_gt[item] != 0):
                return_dict[item] = "Correct";
            elif(dict_infered[item] != 0)and (dict_gt[item] == 0):
                return_dict[item] = "Incorrect";
            elif(dict_infered[item] == 0)and (dict_gt[item] == 0):
                return_dict[item] = "Correct";
            else:
                return_dict[item] = "Missed";
        else:
            return_dict[item] = "Unlabeled";
    return return_dict;

def main():
    args = parse_args()
    
    ext_list = args.exts.split(',')
    ext_list = [ext.lower() for ext in ext_list]
    
    file_list = scan_files(args.image_path, ext_list)
    
    if args.iou_cal_method =='iou':
        function_F = cal_iou
    elif args.iou_cal_method =='min':
        function_F = min_iou
    else:
        raise('iou_cal_method error!');
        
    
    
    det_system = alg_system(args)
    if args.count_num is not None:
        count_box = 0
        count_img = 0
    
    start_time = time.time()
    
    total_count={};
    
    #unhealth_list = ["hsil","lsil", "agc-fn"]
    unhealth_list = args.unhealth_list;
    
    for file_path in file_list:
        img_path = os.path.join(args.image_path, file_path)
        pil_image = Image.open(img_path)
        #pyvips_image = pyvips.Image.new_from_file(img_path)
        print("process {}".format(img_path))
        print("convert pil image to numpy, please wait")
        img = np.asarray(pil_image)
        img = img[:,:, ::-1]
        print("convert success")

        regions = get_regions(img.shape , args.crop_size, args.overlap)
        result = init_dict_box(args.CLASSES_NAME[1:])
        
        for k in range(regions.shape[0]):
            x1, y1, x2, y2 = int(regions[k, 0]), int(regions[k, 1]), \
                             int(regions[k, 2]), int(regions[k, 3])
            patch = img[y1: y2, x1: x2, :].copy()
            #pyvips_patch = pyvips_image.extract_area(x1,y1, x2-x1, y2-y1)
            #patch = vips2numpy(pyvips_patch)
            #patch = patch[:,:, ::-1]
            #tmp_result = det_system.infer_img(patch)
            tmp_result = det_system.infer_image(patch)
            tmp_result = correct_result(tmp_result,[x1,y1,x1,y1])
            
            result = update_result(result, tmp_result)
            #print("Progress: {}/{}".format(str(k), str(regions.shape[0])))

        result = nms_in_class(result)
        result = nms_between_classes(result)
        
        if args.count_num is not None:
          #added_num = count_obj(result)
          added_num = count_unhealth_obj(result, unhealth_list)
          if added_num != 0:
              count_img += 1
          count_box += added_num
          
        before_draw_cost_time = time.time() - start_time;
        print("no draw cost:", before_draw_cost_time);
          
        if args.draw_image:
            img = draw_image(img, result)
            #pyvips_image = draw_vips_image(pyvips_image,result )
        
        #output_path_txt = os.path.join(args.output_path, file_path+".txt")
        single_count_dict =  count_result(result)
        single_count_dict['total'] = 0;
        for item in unhealth_list:
            if item in single_count_dict:
                single_count_dict['total'] +=single_count_dict[item];
        
        
        print(file_path, " Forward result:", single_count_dict);
        for item in single_count_dict:
            if item not in total_count:
                total_count[item] = single_count_dict[item];
            else:
                total_count[item] += single_count_dict[item];
        
#       import pickle
#        with open('result.pickle', 'wb') as handle:
#            pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
#        f=open('result.pickle','rb')
#        result = pickle.load(f)
        
        if args.xml_path is not None:
            
            gt_result = read_xml(os.path.join(args.xml_path, os.path.splitext(file_path)[0]+".xml"), args.CLASSES_NAME)
            gt_count_result = count_result(gt_result)
            find_diff, gt_only_obj_box, model_only_obj_box, same_boxes = compare_result(result, gt_result, args.CLASSES_NAME, function_F)
            print(gt_only_obj_box)
            gt_only_obj = count_result(gt_only_obj_box);
            model_only_obj = count_result(model_only_obj_box);
            
            same_boxes_result_count = count_result(same_boxes)
            if find_diff:
                del gt_only_obj['__background__']
                del model_only_obj['__background__']
                print(file_path," Missed: ", gt_only_obj)
                print(file_path," Incorrect: ", model_only_obj)
            
            print(file_path," Ground_truth: ", gt_count_result)
            print(file_path," Same boxes: ", same_boxes_result_count)
            picture_result_dict = picture_diff_dict(result,gt_result );
            print(file_path," picture: ",picture_result_dict) 
            
            #img = draw_image(img, gt_result, (0,255,0),(128,255,128))
            img = draw_image(img, gt_only_obj_box, (255,64,64))
            
            img = draw_image(img, model_only_obj_box,(64,255,64))
            
            #pyvips_image = draw_vips_image(pyvips_image, gt_only_obj_box, (255,64,64) )
            #pyvips_image = draw_vips_image(pyvips_image, model_only_obj_box,(64,255,64))
            
        output_path = os.path.join(args.output_path, os.path.splitext(file_path)[0]+".jpg")
        if not os.path.isdir(os.path.dirname(output_path)):
          os.makedirs(os.path.dirname(output_path))
        
        if args.draw_image:
            cv2.imwrite(output_path, img)
            #pyvips_image.jpegsave(output_path, Q=50)

    
    cost_time = time.time() - start_time

    if args.count_num is not None:
        total_count['total']  = 0;
        for item in unhealth_list:
            if item in total_count:
                total_count['total'] +=total_count[item];
        print('total count:', total_count);
        #print("total number of boxes is {}".format(str(count_box)))
        #print("total number of images should be checked is {}".format(str(count_img)))
        #print("total number of images is {}".format(str(len(file_list))))
    print("cost time is {}".format(str(cost_time)))
    print("avg time is {}".format(str(cost_time/len(file_list))))
        
    
if __name__ == '__main__':
    main()   
