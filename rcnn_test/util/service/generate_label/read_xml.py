import os
import xml.etree.ElementTree


def init_dict_box(classes):
    new_dict = dict()
    for cls in classes:
        new_dict[cls] = []
    return new_dict


def read_xml(args, file_path):
    xml_path = os.path.join(args.xml_path, os.path.splitext(file_path)[0] + ".xml")
    et = xml.etree.ElementTree.parse(xml_path)

    obj_dict = init_dict_box(args.CLASSES_NAME[1:])
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
