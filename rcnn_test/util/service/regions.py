import numpy as np

from util.service.time_decorator import spend_time


@spend_time
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

    regions = np.array(regions, dtype=np.float32)
    return regions
