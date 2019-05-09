import numpy as np

from util.service.time_decorator import spend_time


@spend_time
def infer_boxes(box_dict, regions, pyvips_image, det_system):
    for k in range(regions.shape[0]):
        x1, y1, x2, y2 = int(regions[k, 0]), int(regions[k, 1]), \
                         int(regions[k, 2]), int(regions[k, 3])
        pyvips_patch = pyvips_image.extract_area(x1, y1, x2 - x1, y2 - y1)
        patch = vips2numpy(pyvips_patch)
        patch = patch[:, :, ::-1]
        tmp_result = det_system.infer_image(patch)
        tmp_result = correct_result(tmp_result, [x1, y1, x1, y1])
        box_dict = update_result(box_dict, tmp_result)
    return box_dict


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


# vips image to numpy array
def vips2numpy(vi):
    return np.ndarray(buffer=vi.write_to_memory(),
                      dtype=format_to_dtype[vi.format],
                      shape=[vi.height, vi.width, vi.bands])


def correct_result(tmp_result, bias_value):
    for k in tmp_result:
        for i in range(len(tmp_result[k])):
            tmp_result[k][i][:4] = [x + y for x, y in zip(tmp_result[k][i][:4], bias_value)]
    return tmp_result


def update_result(result, tmp_result):
    for k in result:
        result[k].extend(tmp_result[k])
    return result
