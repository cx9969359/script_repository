import os
from xml.dom.minidom import Document

from util.service.time_decorator import spend_time


@spend_time
def generate_label_file(result, size, output_label_path, file_path, sub_dir="whole_label"):
    xml_content = generate_whole_xml_contents(result, size)
    output_label_xml_path = os.path.join(output_label_path, sub_dir)

    single_xml_output = os.path.splitext(file_path)[0] + ".xml"
    single_xml_output = os.path.join(output_label_xml_path, single_xml_output)

    if not os.path.isdir(os.path.dirname(single_xml_output)):
        os.makedirs(os.path.dirname(single_xml_output))

    with open(single_xml_output, 'w') as f:
        f.write(xml_content.toprettyxml(indent='        '))


@spend_time
def generate_whole_xml_contents(region_dict, img_size):
    doc = Document()
    anno = doc.createElement('annotation')
    doc.appendChild(anno)
    filename = doc.createElement('filename')
    filename.appendChild(doc.createTextNode("infer_file"))
    anno.appendChild(filename)

    size = doc.createElement('size')
    size.appendChild(doc.createElement('width')).appendChild(doc.createTextNode(str(int(img_size[0]))))
    size.appendChild(doc.createElement('height')).appendChild(doc.createTextNode(str(int(img_size[1]))))

    size.appendChild(doc.createElement('depth')).appendChild(doc.createTextNode(str(3)))
    anno.appendChild(size)

    for label_name in region_dict:
        for box in region_dict[label_name]:
            obj = doc.createElement('object')
            anno.appendChild(obj)
            obj.appendChild(doc.createElement('name')).appendChild(doc.createTextNode(label_name))
            obj.appendChild(doc.createElement('difficult')).appendChild(doc.createTextNode(str(0)))
            obj.appendChild(doc.createElement('truncated')).appendChild(doc.createTextNode(str(0)))
            obj.appendChild(doc.createElement('confidence')).appendChild(doc.createTextNode(str(box[4])))

            bndbox = doc.createElement('bndbox')
            bndbox.appendChild(doc.createElement('xmin')) \
                .appendChild(doc.createTextNode(str(int(box[0]))))
            bndbox.appendChild(doc.createElement('ymin')) \
                .appendChild(doc.createTextNode(str(int(box[1]))))
            bndbox.appendChild(doc.createElement('xmax')) \
                .appendChild(doc.createTextNode(str(int(box[2]))))
            bndbox.appendChild(doc.createElement('ymax')) \
                .appendChild(doc.createTextNode(str(int(box[3]))))
            obj.appendChild(bndbox)

            seg = doc.createElement('segmentation')

            points1 = doc.createElement('points')
            points1.appendChild(doc.createElement('x')) \
                .appendChild(doc.createTextNode(str(int(box[0]))))
            points1.appendChild(doc.createElement('y')) \
                .appendChild(doc.createTextNode(str(int(box[1]))))
            seg.appendChild(points1)

            points2 = doc.createElement('points')
            points2.appendChild(doc.createElement('x')) \
                .appendChild(doc.createTextNode(str(int(box[2]))))
            points2.appendChild(doc.createElement('y')) \
                .appendChild(doc.createTextNode(str(int(box[1]))))
            seg.appendChild(points2)

            points3 = doc.createElement('points')
            points3.appendChild(doc.createElement('x')) \
                .appendChild(doc.createTextNode(str(int(box[2]))))
            points3.appendChild(doc.createElement('y')) \
                .appendChild(doc.createTextNode(str(int(box[3]))))
            seg.appendChild(points3)

            points4 = doc.createElement('points')
            points4.appendChild(doc.createElement('x')) \
                .appendChild(doc.createTextNode(str(int(box[0]))))
            points4.appendChild(doc.createElement('y')) \
                .appendChild(doc.createTextNode(str(int(box[3]))))
            seg.appendChild(points4)

            obj.appendChild(seg)

    return doc
