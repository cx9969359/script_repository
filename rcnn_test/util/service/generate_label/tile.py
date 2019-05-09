import os
from xml.dom.minidom import Document

from util.service.time_decorator import spend_time


@spend_time
def generate_label_files(box_dict, regions, output_label_path, file_path, pyvips_image, sub_img_dir="img",
                         sub_label_dir="label"):
    crop_region_dict = get_cropped_region(box_dict, regions)

    xml_contents, regions_index = generate_xml_contents(crop_region_dict, regions)

    output_label_img_path = os.path.join(output_label_path, sub_img_dir)
    output_label_xml_path = os.path.join(output_label_path, sub_label_dir)

    # write files
    crop_img_counter = 0
    for xml_doc, reg_index in zip(xml_contents, regions_index):

        single_img_output = os.path.splitext(file_path)[0] + "_cropped_{:04d}".format(crop_img_counter) + ".png"
        single_xml_output = os.path.splitext(file_path)[0] + "_cropped_{:04d}".format(crop_img_counter) + ".xml"

        single_img_output = os.path.join(output_label_img_path, single_img_output)
        single_xml_output = os.path.join(output_label_xml_path, single_xml_output)

        if not os.path.isdir(os.path.dirname(single_img_output)):
            os.makedirs(os.path.dirname(single_img_output))

        if not os.path.isdir(os.path.dirname(single_xml_output)):
            os.makedirs(os.path.dirname(single_xml_output))

        # image part
        tmp_pyvips_img = pyvips_image.extract_area(regions[reg_index, 0], regions[reg_index, 1],
                                                   regions[reg_index, 2] - regions[reg_index, 0],
                                                   regions[reg_index, 3] - regions[reg_index, 1])
        tmp_pyvips_img.pngsave(single_img_output, Q=100)

        # xml part
        with open(single_xml_output, 'w') as f:
            f.write(xml_doc.toprettyxml(indent='        '))
        crop_img_counter += 1


# result format:
# {label_name:[[x1,y1,x2,y2, conf]]}
# regions format: numpy,  [x1, y1, x2, y2]
def get_cropped_region(result, regions):
    regions_box = dict()

    for index in range(regions.shape[0]):

        x1 = regions[index, 0]
        y1 = regions[index, 1]
        x2 = regions[index, 2]
        y2 = regions[index, 3]

        for label_name in result:
            for box in result[label_name]:
                if box[0] >= x1 and box[1] >= y1 and box[2] <= x2 and box[3] <= y2:
                    if index not in regions_box:
                        regions_box[index] = [[box[0], box[1], box[2], box[3], label_name, box[4]]]
                    else:
                        regions_box[index].append([box[0], box[1], box[2], box[3], label_name, box[4]])

    return regions_box


@spend_time
def generate_xml_contents(crop_region_dict, regions):
    xml_content_list = []
    regions_index_list = []

    for region_index in crop_region_dict:
        x1 = regions[region_index, 0]
        y1 = regions[region_index, 1]
        x2 = regions[region_index, 2]
        y2 = regions[region_index, 3]

        boxes_list = crop_region_dict[region_index]

        # xml create
        doc = Document()
        anno = doc.createElement('annotation')
        doc.appendChild(anno)
        filename = doc.createElement('filename')
        filename.appendChild(doc.createTextNode(str(region_index - 1).zfill(8)))
        anno.appendChild(filename)
        relative = doc.createElement('relative')
        relative.appendChild(doc.createElement('x')). \
            appendChild(doc.createTextNode(str(int(x1))))
        relative.appendChild(doc.createElement('y')). \
            appendChild(doc.createTextNode(str(int(y1))))
        relative.appendChild(doc.createElement('rawpath')). \
            appendChild(doc.createTextNode("cropped"))
        anno.appendChild(relative)

        size = doc.createElement('size')
        size.appendChild(doc.createElement('width')).appendChild(doc.createTextNode(str(int(x2 - x1))))
        size.appendChild(doc.createElement('height')).appendChild(doc.createTextNode(str(int(y2 - y1))))

        size.appendChild(doc.createElement('depth')).appendChild(doc.createTextNode(str(3)))
        anno.appendChild(size)

        for box_content in boxes_list:
            obj = doc.createElement('object')
            anno.appendChild(obj)
            obj.appendChild(doc.createElement('name')).appendChild(doc.createTextNode(box_content[4]))
            diff = doc.createElement('difficult')
            diff.appendChild(doc.createTextNode(str(0)))
            trun = doc.createElement('truncated')
            trun.appendChild(doc.createTextNode(str(0)))
            obj.appendChild(diff)
            obj.appendChild(trun)
            bndbox = doc.createElement('bndbox')
            bndbox.appendChild(doc.createElement('xmin')) \
                .appendChild(doc.createTextNode(str(int(box_content[0] - x1))))
            bndbox.appendChild(doc.createElement('ymin')) \
                .appendChild(doc.createTextNode(str(int(box_content[1] - y1))))
            bndbox.appendChild(doc.createElement('xmax')) \
                .appendChild(doc.createTextNode(str(int(box_content[2] - x1))))
            bndbox.appendChild(doc.createElement('ymax')) \
                .appendChild(doc.createTextNode(str(int(box_content[3] - y1))))
            obj.appendChild(bndbox)

            x_min = str(int(box_content[0] - x1))
            y_min = str(int(box_content[1] - y1))
            x_max = str(int(box_content[2] - x1))
            y_max = str(int(box_content[3] - y1))
            segmentation = doc.createElement('segmentation')

            points = doc.createElement('points')
            points.appendChild(doc.createElement('x')).appendChild(doc.createTextNode(x_min))
            points.appendChild(doc.createElement('y')).appendChild(doc.createTextNode(y_min))
            segmentation.appendChild(points)

            points = doc.createElement('points')
            points.appendChild(doc.createElement('x')).appendChild(doc.createTextNode(x_min))
            points.appendChild(doc.createElement('y')).appendChild(doc.createTextNode(y_max))
            segmentation.appendChild(points)

            points = doc.createElement('points')
            points.appendChild(doc.createElement('x')).appendChild(doc.createTextNode(x_max))
            points.appendChild(doc.createElement('y')).appendChild(doc.createTextNode(y_min))
            segmentation.appendChild(points)

            points = doc.createElement('points')
            points.appendChild(doc.createElement('x')).appendChild(doc.createTextNode(x_max))
            points.appendChild(doc.createElement('y')).appendChild(doc.createTextNode(y_max))
            segmentation.appendChild(points)

            obj.appendChild(segmentation)

        xml_content_list.append(doc)
        regions_index_list.append(region_index)

    return xml_content_list, regions_index_list
