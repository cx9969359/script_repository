import os
import xml.etree.ElementTree as ET
from xml.dom.minidom import Document

# PIL not used in this code
import PIL
from PIL import Image
# PIL 主要用于生成seg图片
import pyvips
import numpy as np
import scipy.stats as st
import imutils
import cv2
from multiprocessing import Pool, cpu_count
from functools import partial

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


def vips2numpy(vi):
    return np.ndarray(buffer=vi.write_to_memory(),
                      dtype=format_to_dtype[vi.format],
                      shape=[vi.height, vi.width, vi.bands])


def scan_files_and_create_folder(input_file_path, output_seg_path, ext_list):
    file_list = []

    for root, dirs, files in os.walk(input_file_path):
        # create folder if it does not exist
        new_path = os.path.join(root, '').replace('\\', '/').replace(
            os.path.join(input_file_path, '').replace('\\', '/'), os.path.join(output_seg_path, '').replace('\\', '/'),
            1)
        # print('#### debug new_path')
        # print(new_path)
        if not os.path.exists(new_path):
            os.makedirs(new_path)

        # scan all files and put it into list
        for f in files:
            if os.path.splitext(f)[1].lower() in ext_list:
                file_list.append(os.path.splitext(os.path.join(root, f).replace('\\', '/').replace(
                    os.path.join(input_file_path, '').replace('\\', '/'), ''))[0])

    return file_list


def get_image_shape(xml_tree):
    height = int(xml_tree.find('size').find('height').text)
    width = int(xml_tree.find('size').find('width').text)
    return (height, width)


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


def get_regions_record(xml_tree, doing_list, ignore_list, regions, crop_size, crop_threshold=0., file_postfix='',
                       zfill_value=0):
    objects = xml_tree.findall('object')
    pool = Pool(cpu_count())
    _get_record = partial(get_record, doing_list=doing_list, ignore_list=ignore_list, objects=objects,
                          crop_threshold=crop_threshold, file_postfix=file_postfix,
                          zfill_value=zfill_value, crop_size=crop_size)
    regions_record = pool.map(_get_record, regions)
    regions_record = set(regions_record)
    return regions_record


def get_record(region, doing_list, ignore_list, objects, crop_threshold, file_postfix, zfill_value, crop_size):
    obj_index_list = []
    for index, obj in enumerate(objects):
        cls_name = obj.find('name').text.lower().strip()
        if len(ignore_list) != 0 and cls_name in ignore_list:
            continue
        elif len(doing_list) != 0 and (cls_name not in doing_list):
            continue
        obj_index_list.append(index)

    x1, y1, x2, y2 = region
    keep = []
    ls = 0
    rs = 0
    us = 0
    ds = 0
    for index in obj_index_list:
        bbox = objects[index].find('bndbox')
        bx1 = float(bbox.find('xmin').text)
        by1 = float(bbox.find('ymin').text)
        bx2 = float(bbox.find('xmax').text)
        by2 = float(bbox.find('ymax').text)

        ix1 = max(bx1, x1)
        iy1 = max(by1, y1)
        ix2 = min(bx2, x2)
        iy2 = min(by2, y2)

        w = max(0, ix2 - ix1)
        h = max(0, iy2 - iy1)
        inter = w * h

        if inter > 0:
            ls = int(max(max((x1 - bx1), 0), ls))
            rs = int(max(max((bx2 - x2), 0), rs))
            us = int(max(max((y1 - by1), 0), us))
            ds = int(max(max((by2 - y2), 0), ds))

            keep.append(index)

    # base array, fill 1 for each object
    base_arr = np.zeros((us + crop_size + ds, ls + crop_size + rs), dtype=np.float32)
    mask_arr = np.zeros((us + crop_size + ds, ls + crop_size + rs), dtype=np.float32)

    # draw 1 for all obj in current region
    single_points_array_list = []
    for index in keep:
        points_arr = get_all_points(objects[index])
        # rot_box: [cen_x, cen_y, width, height, angle]
        rot_box = get_rot_box_info(objects[index])
        if objects[index].find('name').text.lower().strip() == 'hsil' or \
                objects[index].find('name').text.lower().strip() == 'scc':
            sv = 0.76
            ev = 1.
        else:
            sv = 0.51
            ev = 0.75
        # print(rot_box)
        kernel = get_kern(sv, ev, rot_box[2], rot_box[3], rot_box[4])

        points_arr = correct_points(points_arr, x1 - ls, y1 - us)

        single_points_array_list.append(points_arr)

        cen_points = correct_points(np.asarray(rot_box[:2]), x1 - ls, y1 - us)
        mask_arr = fill_by_points(points_arr, mask_arr)
        base_arr = fill_seg_value_by_points(base_arr, mask_arr, kernel, cen_points[0], cen_points[1])

    if (np.sum(mask_arr[us: us + crop_size, ls:ls + crop_size]) / (crop_size * crop_size)) <= crop_threshold:
        return
    print('generate regions_record')
    patch_idx = 0
    result = '{:d},{:d},{:d},{:d},{}'.format(x1, y1, x2, y2,
                                             '_' + file_postfix + '_' + str(patch_idx).zfill(zfill_value))
    result = '{},{},{},{},{}'.format(x1, y1, x2, y2, single_points_array_list)
    return result


# rotation anticlockwise
def get_kern(sv, ev, xkernlen=20, ykernlen=10, rotate=0, xnsig=3.5, ynsig=3.5):
    '''Returns a 2D Gaussian kernel array.'''

    interval_x = (2 * xnsig + 1.) / (xkernlen)
    interval_y = (2 * ynsig + 1.) / (ykernlen)

    x = np.linspace(-xnsig - interval_x / 2., xnsig + interval_x / 2., xkernlen + 1)
    y = np.linspace(-ynsig - interval_y / 2., ynsig + interval_y / 2., ykernlen + 1)

    kern1d_x = np.diff(st.norm.cdf(x))

    kern1d_y = np.diff(st.norm.cdf(y))

    kernel_raw = np.sqrt(np.outer(kern1d_y, kern1d_x))
    kernel = kernel_raw / kernel_raw.sum()
    # print(kernel[int(ykernlen/2), int(xkernlen/2)])

    # change all value in kernel
    range_value = abs(ev - sv)
    kernel = kernel * (range_value / kernel[int(ykernlen / 2), int(xkernlen / 2)])
    if sv < ev:
        symbol = 1.
        base = sv
    else:
        symbol = -1.
        base = ev

    size = max(kernel.shape)
    start_x = int(size / 2) - int(xkernlen / 2)
    start_y = int(size / 2) - int(ykernlen / 2)
    # end_x = int(size/2)+int(xkernlen/2)
    # end_y = int(size/2)+int(ykernlen/2)
    end_x = start_x + xkernlen
    end_y = start_y + ykernlen

    tmp_kernel = np.zeros((size, size))
    tmp_kernel[start_y:end_y, start_x:end_x] = kernel

    kernel = imutils.rotate(tmp_kernel, rotate)

    kernel = kernel * symbol + base
    # print('#####kernel_shape_debug')
    # print(kernel.shape)

    return kernel


def fill_by_points(points_arr, img_arr, value=(1)):
    tmp_arr = img_arr.copy()
    pts = points_arr.copy()
    pts = pts.reshape((-1, 1, 2))
    return cv2.fillPoly(tmp_arr, [pts], value)


def fill_seg_value_by_points(base_arr, mask_arr, kernel, cen_x, cen_y):
    height, width = base_arr.shape
    half_kern_size = int(kernel.shape[0] / 2)

    kern_x1 = cen_x - half_kern_size
    kern_y1 = cen_y - half_kern_size
    kern_x2 = kern_x1 + kernel.shape[0]
    kern_y2 = kern_y1 + kernel.shape[0]

    arr_x1 = max(0, kern_x1)
    arr_y1 = max(0, kern_y1)
    arr_x2 = min(width, kern_x2)
    arr_y2 = min(height, kern_y2)

    base_arr[arr_y1:arr_y2, arr_x1:arr_x2] = mask_arr[arr_y1:arr_y2, arr_x1:arr_x2] * \
                                             kernel[arr_y1 - kern_y1: kernel.shape[0] - (kern_y2 - arr_y2),
                                             arr_x1 - kern_x1: kernel.shape[0] - (kern_x2 - arr_x2)]

    # kern_sx, kern_x1 = (0, kern_x1) if kern_x1 >= 0 else (-kern_x1, 0)
    # kern_sy, kern_y1 = (0, kern_y1) if kern_y1 >= 0 else (-kern_y1, 0)
    # kern_ex, kern_x2 = (half_kern_size*2, kern_x2) if kern_x2 <= width else (half_kern_size*2 - (kern_x2-width), width)
    # kern_ey, kern_y2 = (half_kern_size*2, kern_y2) if kern_y2 <= height else (half_kern_size*2 - (kern_y2-height), height)

    # base_arr[kern_y1:kern_y1+kernel.shape[0], kern_x1:kern_x1+kernel.shape[0]] = mask_arr[kern_y1:kern_y1+kernel.shape[0], kern_x1:kern_x1+kernel.shape[0]] * kernel
    # done

    return base_arr


def get_all_points(obj):
    points = []

    for point_obj in obj.find('segmentation').findall('points'):
        points.append(int(float(point_obj.find('x').text)))
        points.append(int(float(point_obj.find('y').text)))

    return np.asarray(points, dtype=np.int32)


def get_rot_box_info(obj):
    # rot_box: [cen_x, cen_y, width, height, angle]
    rot_box = []
    rot_obj = obj.find('rot_box')
    rot_box.append(int(float(rot_obj.find('center_x').text)))
    rot_box.append(int(float(rot_obj.find('center_y').text)))
    rot_box.append(int(float(rot_obj.find('box_width').text)))
    rot_box.append(int(float(rot_obj.find('box_height').text)))
    rot_box.append(abs(float(rot_obj.find('box_angle').text)))
    return rot_box


def correct_points(points_arr, start_x, start_y):
    points_arr[0::2] = points_arr[0::2] - start_x
    points_arr[1::2] = points_arr[1::2] - start_y
    return points_arr


def process_file(file_path, anno_dir, output_dir, crop_size, overlap, crop_threshold, resize_size, file_postfix,
                 ignore_list, doing_list, zfill_value=12):
    anno_path = os.path.join(anno_dir, file_path).replace('\\', '/') + '.xml'

    seg_prefix_path = os.path.join(output_dir, file_path).replace('\\', '/')

    tree = ET.parse(anno_path)
    img_height = int(tree.find('size').find('height').text)
    img_width = int(tree.find('size').find('width').text)

    regions = get_regions([img_height, img_width], crop_size, overlap)

    objs = tree.findall('object')

    obj_index_list = []

    for ix, obj in enumerate(objs):
        cls_name = obj.find('name').text.lower().strip()
        if len(ignore_list) != 0 and cls_name in ignore_list:
            continue
        elif len(doing_list) != 0 and (cls_name not in doing_list):
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

            w = max(0, ix2 - ix1)
            h = max(0, iy2 - iy1)
            inter = w * h

            if inter > 0:
                ls = int(max(max((x1 - bx1), 0), ls))
                rs = int(max(max((bx2 - x2), 0), rs))
                us = int(max(max((y1 - by1), 0), us))
                ds = int(max(max((by2 - y2), 0), ds))

                keep.append(index)

        # base array, fill 1 for each object   
        base_arr = np.zeros((us + crop_size + ds, ls + crop_size + rs), dtype=np.float32)
        mask_arr = np.zeros((us + crop_size + ds, ls + crop_size + rs), dtype=np.float32)

        # draw 1 for all obj in current region
        for index in keep:
            points_arr = get_all_points(objs[index])
            # rot_box: [cen_x, cen_y, width, height, angle]
            rot_box = get_rot_box_info(objs[index])
            if objs[index].find('name').text.lower().strip() == 'hsil' or \
                    objs[index].find('name').text.lower().strip() == 'scc':
                sv = 0.76
                ev = 1.
            else:
                sv = 0.51
                ev = 0.75
            # print(rot_box)
            kernel = get_kern(sv, ev, rot_box[2], rot_box[3], rot_box[4])

            points_arr = correct_points(points_arr, x1 - ls, y1 - us)

            cen_points = correct_points(np.asarray(rot_box[:2]), x1 - ls, y1 - us)
            mask_arr = fill_by_points(points_arr, mask_arr)
            base_arr = fill_seg_value_by_points(base_arr, mask_arr, kernel, cen_points[0], cen_points[1])

        # print('### debug area check')
        # print(np.sum(mask_arr[us: us+crop_size, ls:ls+crop_size])/(crop_size*crop_size))

        if (np.sum(mask_arr[us: us + crop_size, ls:ls + crop_size]) / (crop_size * crop_size)) <= crop_threshold:
            continue
        # base_arr = base_arr + 0.5
        # output_seg_path = seg_prefix_path + '_' + file_postfix + '_' + str(patch_idx).zfill(zfill_value) + '.npy'
        # print('### debug try to create')
        # print(output_seg_path)
        print('create file')
        # np.save(output_seg_path, base_arr[us: us + crop_size, ls:ls + crop_size])
        regions_record.append('{:d},{:d},{:d},{:d},{}'.format(x1, y1, x2, y2,
                                                              '_' + file_postfix + '_' + str(patch_idx).zfill(
                                                                  zfill_value)))
        patch_idx += 1
    return regions_record
    # with open(seg_prefix_path + '_record.txt', 'w') as f:
    #     for r in regions_record:
    #         f.write(r + '\n')


def create_seg(anno_dir, output_dir, crop_size=900, overlap=300, crop_threshold=0., resize_size=0, file_postfix='',
               ignore_list=[], doing_list=[], num_process=1):
    anno_dir = os.path.join(anno_dir, '').replace('\\', '/')
    output_dir = os.path.join(output_dir, '').replace('\\', '/')

    assert not (len(ignore_list) != 0 and len(doing_list) != 0)

    file_list = scan_files_and_create_folder(anno_dir, output_dir, ['.xml'])

    regions_record_list = []
    file_point_array_list = []
    for anno_file in file_list:
        regions_record, point_array_list = process_file(anno_file, anno_dir, output_dir, crop_size, overlap,
                                                        crop_threshold, resize_size,
                                                        file_postfix,
                                                        ignore_list, doing_list)
        regions_record_list.append(regions_record)
        file_point_array_list.append(point_array_list)
    return regions_record_list, file_point_array_list


def read_palette_file(palette_file_path):
    palette_list = [0] * (256 * 3)
    label_list = [''] * 256
    counter = 0
    with open(palette_file_path) as f:
        for line in f:
            line = line.rstrip()
            palette_list[counter:counter + 3] = map(int, line.split(' ')[:3])
            label_list[int(counter / 3)] = line.split(' ')[3:]
            counter += 3
            if counter >= (256 * 3):
                break

    if counter < 256 * 3:
        palette_list[-3:] = palette_list[counter - 3:counter]
        palette_list[counter - 3:counter] = [0, 0, 0]
        label_list[-1] = label_list[int((counter - 3) / 3)]
        label_list[int((counter - 3) / 3)] = ''

    return palette_list, label_list


def create_contour_image(img_np, all_shapes, label_shape_dict, label_list, init_type):
    if init_type == 0:
        img_mat = np.full((img_np.shape[0], img_np.shape[1], 1), float(0))
    else:
        img_mat = np.full((img_np.shape[0], img_np.shape[1], 1), float(ignore_label_index))

    # for i in range(len(label_list)-1):
    #     for j in range(len(label_list[i])):
    #         for x in label_shape_dict[label_list[i][j]]:
    #             contour = np.asarray(all_shapes[tmp_shape_index]['points'])

    for tmp_color_index in range(len(label_list) - 1):
        for tmp_label_index in range(len(label_list[tmp_color_index])):
            for tmp_shape_index in label_shape_dict[label_list[tmp_color_index][tmp_label_index]]:
                contour = np.asarray(all_shapes[tmp_shape_index]['points'])
                draw_single_contour_to_image(img_mat, contour, (tmp_color_index), (ignore_label_index))

    return img_mat


def get_label_indexes(file_point_array_list, label_string):
    all_shapes = file_point_array_list
    all_shapes_len = len(all_shapes)
    index_list = []
    for num in range(0, all_shapes_len):
        if all_shapes[num]['label'] == label_string:
            index_list.append(num)

    return index_list


def write_single_image(input_json_file, file_point_array_list, dest_path, palette_list, label_list, init_type,
                       regions_record_list):
    label_shape_dict = {}
    for tmp_color_index in range(len(label_list)):
        for tmp_label_index in range(len(label_list[tmp_color_index])):
            label_shape_dict[label_list[tmp_color_index][tmp_label_index]] = get_label_indexes(file_point_array_list,
                                                                                               label_list[
                                                                                                   tmp_color_index][
                                                                                                   tmp_label_index])
    all_shapes = json_data['shapes']

    # 切图
    for region in regions_record_list[0]:
        region = [int(region.split(',')[0]), int(region.split(',')[1]), int(region.split(',')[2]),
                  int(region.split(',')[3])]
        image_path = 'F:\\tif_images\\thyroid\\more.tif'
        pyvips_image = pyvips.Image.new_from_file(image_path)
        patch = pyvips_image.extract_area(region[0], region[1], region[2], region[3])
        img_np = vips2numpy(patch)

        output_mat = create_contour_image(img_np, all_shapes, label_shape_dict, label_list, init_type)

        check_mat = np.full(output_mat.shape, float(ignore_label_index))

        if np.array_equal(output_mat, check_mat):
            print('all label ignored for file {}'.format(input_json_file))
        else:
            pil_mat = np.squeeze(output_mat, axis=2)
            pil_img = Image.fromarray(pil_mat)
            pil_img = pil_img.convert('P')
            pil_img.putpalette(palette_list)
            pil_img.save(dest_path)


if __name__ == '__main__':
    annotation_xml_path = '././label_xml/V201803956LSIL_2019_01_28_15_26_39.xml'
    output_dir = '././out_put_png'
    crop_size = 900
    overlap = 300
    doing_list = ['hsil', 'scc', 'lsil']
    ignore_list = []
    annotation_xml_tree = ET.parse(annotation_xml_path)

    image_shape = get_image_shape(annotation_xml_tree)
    regions = get_regions(image_shape, crop_size, overlap)

    regions_record = get_regions_record(annotation_xml_tree, doing_list, ignore_list, regions, crop_size,
                                        crop_threshold=0.,
                                        file_postfix='cropped', zfill_value=12)
    print(regions_record)
    for i in regions_record:
        print(i)
    # regions_record_list, file_point_array_list = create_seg(anno_dir, output_dir, file_postfix='cropped',
    #                                                         doing_list=doing_list)

    # print(file_point_array_list)
    # print(len(file_point_array_list))
    # input_palette = 'F:/split_image/palette_folder/palette.txt'
    # palette_list, label_list = read_palette_file(input_palette)
    #
    # dest_path = 'F:/split_image/image/a.png'
    # init_type = 0
    # input_json_file = ''
    # # dest_path = re.sub(os.path.join(input_json_path, ''), os.path.join(output_png_path, ''), (os.path.splitext(json_file)[0])) + '.png'
    # write_single_image(input_json_file, file_point_array_list, dest_path, palette_list, label_list, init_type,regions_record_list)
