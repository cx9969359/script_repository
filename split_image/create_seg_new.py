import os
import xml.etree.ElementTree as ET
from xml.dom.minidom import Document

import PIL
from PIL import Image
# PIL 主要用于生成seg图片

import numpy as np
import scipy.stats as st
import imutils
import cv2


def scan_files_and_create_folder(input_file_path, output_seg_path, ext_list):
    file_list = []

    for root, dirs, files in os.walk(input_file_path):
        # create folder if it does not exist
        new_path = os.path.join(root,"").replace("\\","/").replace(os.path.join(input_file_path, "").replace("\\","/"), os.path.join(output_seg_path, "").replace("\\","/"),1)
        #print("#### debug new_path")
        #print(new_path)
        if not os.path.exists(new_path):
            os.makedirs(new_path)
            

        # scan all files and put it into list
        for f in files:
            if os.path.splitext(f)[1].lower() in ext_list:
                file_list.append(os.path.splitext(os.path.join(root,f).replace("\\","/").replace(os.path.join(input_file_path, "").replace("\\","/"), ""))[0])

    return file_list
    
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

    return regions

    
# rotation anticlockwise
def get_kern(sv, ev, xkernlen=20, ykernlen=10, rotate=0, xnsig=3.5, ynsig=3.5):
    """Returns a 2D Gaussian kernel array."""

    interval_x = (2*xnsig+1.)/(xkernlen)
    interval_y = (2*ynsig+1.)/(ykernlen)
    
    x = np.linspace(-xnsig-interval_x/2., xnsig+interval_x/2., xkernlen+1)
    y = np.linspace(-ynsig-interval_y/2., ynsig+interval_y/2., ykernlen+1)
    
    kern1d_x = np.diff(st.norm.cdf(x))
 
    kern1d_y = np.diff(st.norm.cdf(y))

    kernel_raw = np.sqrt(np.outer(kern1d_y, kern1d_x))
    kernel = kernel_raw/kernel_raw.sum()
    #print(kernel[int(ykernlen/2), int(xkernlen/2)])
    
    
    # change all value in kernel
    range_value = abs(ev-sv)
    kernel = kernel * (range_value/kernel[int(ykernlen/2), int(xkernlen/2)])
    if sv < ev:
        symbol = 1.
        base = sv
    else:
        symbol = -1.
        base = ev
    
    size = max(kernel.shape)    
    start_x = int(size/2)-int(xkernlen/2)
    start_y = int(size/2)-int(ykernlen/2)
    #end_x = int(size/2)+int(xkernlen/2)
    #end_y = int(size/2)+int(ykernlen/2)
    end_x = start_x + xkernlen
    end_y = start_y + ykernlen
    
    tmp_kernel = np.zeros((size,size))
    tmp_kernel[start_y:end_y, start_x:end_x] = kernel
    
    kernel = imutils.rotate(tmp_kernel, rotate)
    
    kernel = kernel * symbol + base
    #print("#####kernel_shape_debug")
    #print(kernel.shape)
    
    return kernel
 
def fill_by_points(points_arr, img_arr, value=(1)):
    tmp_arr = img_arr.copy()
    pts = points_arr.copy()
    pts = pts.reshape((-1,1,2))
    return cv2.fillPoly(tmp_arr, [pts], value)
    
def fill_seg_value_by_points(base_arr, mask_arr, kernel, cen_x, cen_y):
    height, width = base_arr.shape
    half_kern_size = int(kernel.shape[0]/2)
    
    kern_x1 = cen_x - half_kern_size
    kern_y1 = cen_y - half_kern_size
    kern_x2 = kern_x1 + kernel.shape[0]
    kern_y2 = kern_y1 + kernel.shape[0]
    
    arr_x1 = max(0, kern_x1)
    arr_y1 = max(0, kern_y1)
    arr_x2 = min(width, kern_x2)
    arr_y2 = min(height, kern_y2)
    
    base_arr[arr_y1:arr_y2, arr_x1:arr_x2] = mask_arr[arr_y1:arr_y2, arr_x1:arr_x2] * \
                              kernel[arr_y1-kern_y1: kernel.shape[0]-(kern_y2-arr_y2), arr_x1-kern_x1: kernel.shape[0]-(kern_x2-arr_x2)]
    

    #kern_sx, kern_x1 = (0, kern_x1) if kern_x1 >= 0 else (-kern_x1, 0)
    #kern_sy, kern_y1 = (0, kern_y1) if kern_y1 >= 0 else (-kern_y1, 0)
    #kern_ex, kern_x2 = (half_kern_size*2, kern_x2) if kern_x2 <= width else (half_kern_size*2 - (kern_x2-width), width)
    #kern_ey, kern_y2 = (half_kern_size*2, kern_y2) if kern_y2 <= height else (half_kern_size*2 - (kern_y2-height), height)
   
    #base_arr[kern_y1:kern_y1+kernel.shape[0], kern_x1:kern_x1+kernel.shape[0]] = mask_arr[kern_y1:kern_y1+kernel.shape[0], kern_x1:kern_x1+kernel.shape[0]] * kernel
    # done
    
    return base_arr
    
def get_all_points(obj):
    points = []
    
    for point_obj in obj.find("segmentation").findall("points"):
        points.append(int(float(point_obj.find("x").text)))
        points.append(int(float(point_obj.find("y").text)))
    
    return np.asarray(points, dtype=np.int32)
    
def get_rot_box_info(obj):
    # rot_box: [cen_x, cen_y, width, height, angle]
    rot_box = []
    rot_obj = obj.find("rot_box")
    rot_box.append(int(float(rot_obj.find("center_x").text)))
    rot_box.append(int(float(rot_obj.find("center_y").text)))
    rot_box.append(int(float(rot_obj.find("box_width").text)))
    rot_box.append(int(float(rot_obj.find("box_height").text)))
    rot_box.append(abs(float(rot_obj.find("box_angle").text)))
    return rot_box
    
def correct_points(points_arr, start_x, start_y):
    points_arr[0::2] = points_arr[0::2] - start_x
    points_arr[1::2] = points_arr[1::2] - start_y
    return points_arr

    
def process_file(file_path, anno_dir, output_dir, crop_size, overlap, crop_threshold, resize_size, file_postfix, ignore_list, doing_list, zfill_value=12):
    anno_path = os.path.join(anno_dir, file_path).replace("\\","/") + ".xml"
    
    seg_prefix_path = os.path.join(output_dir,file_path).replace("\\","/")
    #print("####debug")
    #print(output_dir)
    #print(file_path)
    
    tree = ET.parse(anno_path)
    img_height = int(tree.find("size").find("height").text)
    img_width = int(tree.find("size").find("width").text)
    
    regions = get_regions([img_height, img_width], crop_size, overlap)
    
    objs = tree.findall('object')
    
    obj_index_list = []
    
    for ix, obj in enumerate(objs):
        cls_name = obj.find('name').text.lower().strip()
        if len(ignore_list) != 0 and cls_name in ignore_list:
            continue
        elif len(doing_list) !=0 and (cls_name not in doing_list):
            continue
        obj_index_list.append(ix)
    
    patch_idx = 0
    regions_record = []
    for region in regions:
        x1, y1, x2, y2 = region
        keep = []
        
        ls = 0
        rs = 0
        us = 0
        ds = 0
               
        for index in obj_index_list:
            bbox = objs[index].find('bndbox')
            bx1 = float(bbox.find('xmin').text)
            by1 = float(bbox.find('ymin').text)
            bx2 = float(bbox.find('xmax').text)
            by2 = float(bbox.find('ymax').text)
            
            
            ix1 = max(bx1, x1)
            iy1 = max(by1, y1)
            ix2 = min(bx2, x2)
            iy2 = min(by2, y2)
            #print("####check")
            #print((x1, y1, x2, y2))
            #print((bx1,by1,bx2,by2))
            #print((ix1, iy1, ix2, iy2))
            
            
            w = max(0, ix2 - ix1)
            h = max(0, iy2 - iy1)
            inter = w * h
            
            if inter > 0:
                #print("####check")
                #print((x1, y1, x2, y2))
                #print((bx1,by1,bx2,by2))
                #print((ix1, iy1, ix2, iy2))
                # get extend area value
                ls = int(max(max((x1-bx1),0),ls))
                rs = int(max(max((bx2-x2),0),rs))
                us = int(max(max((y1-by1),0),us))
                ds = int(max(max((by2-y2),0),ds))
                
                keep.append(index)
                
        # base array, fill 1 for each object   

        base_arr = np.zeros((us+crop_size+ds, ls+crop_size+rs), dtype=np.float32)
        mask_arr = np.zeros((us+crop_size+ds, ls+crop_size+rs), dtype=np.float32)
        
        # draw 1 for all obj in current region
        for index in keep:
            points_arr = get_all_points(objs[index])
            # rot_box: [cen_x, cen_y, width, height, angle]
            rot_box = get_rot_box_info(objs[index])
            if objs[index].find("name").text.lower().strip() == "hsil" or \
               objs[index].find("name").text.lower().strip() == "scc":
                sv = 0.76
                ev = 1.
            else:
                sv = 0.51
                ev = 0.75
            #print("##### debug rotbox")
            #print(rot_box)
            kernel = get_kern(sv, ev, rot_box[2], rot_box[3], rot_box[4])
            
            points_arr = correct_points(points_arr, x1-ls, y1-us)
            cen_points = correct_points(np.asarray(rot_box[:2]), x1-ls, y1-us)
            
            mask_arr = fill_by_points(points_arr, mask_arr)
            
            base_arr = fill_seg_value_by_points(base_arr, mask_arr, kernel, cen_points[0], cen_points[1])
        
        
        
        #print("### debug area check")
        #print(np.sum(mask_arr[us: us+crop_size, ls:ls+crop_size])/(crop_size*crop_size))
        
        if (np.sum(mask_arr[us: us+crop_size, ls:ls+crop_size])/(crop_size*crop_size)) <= crop_threshold:
            continue
        #base_arr = base_arr + 0.5
        output_seg_path = seg_prefix_path + "_" + file_postfix + "_" + str(patch_idx).zfill(zfill_value) + '.npy'
        #print("### debug try to create")
        #print(output_seg_path)
        print("create file {}".format(output_seg_path))
        np.save(output_seg_path, base_arr[us: us+crop_size, ls:ls+crop_size])
        regions_record.append("{:d},{:d},{:d},{:d},{}".format(x1, y1, x2, y2,"_" + file_postfix + "_" + str(patch_idx).zfill(zfill_value)))
        patch_idx += 1
    with open(seg_prefix_path + "_record.txt", 'w') as f:
        for r in regions_record:
            f.write(r+"\n")
        
def create_seg(anno_dir, output_dir, crop_size=900, overlap=300, crop_threshold=0., resize_size=0, file_postfix="", ignore_list=[], doing_list=[], num_process=1):
    anno_dir = os.path.join(anno_dir, "").replace("\\","/")
    output_dir = os.path.join(output_dir, "").replace("\\","/")
    
    assert not (len(ignore_list) != 0 and len(doing_list) != 0)
    
    file_list = scan_files_and_create_folder(anno_dir, output_dir, [".xml"])
 
    for anno_file in file_list:
       process_file(anno_file, anno_dir, output_dir, crop_size, overlap, crop_threshold, resize_size, file_postfix, ignore_list, doing_list)
    
    
if __name__ == '__main__':

    anno_dir = 'D:/data_process_center/test_img/test/tmp_label'
    output_dir = 'D:/data_process_center/test_img/test/output'
    
    doing_list = ["hsil","scc","lsil"]
    
    create_seg(anno_dir, output_dir, file_postfix="cropped", doing_list=doing_list)