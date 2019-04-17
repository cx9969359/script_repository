import os
import xml.etree.cElementTree as ET

from flask import request, jsonify
from flask_login import login_required
from flask_restplus import Namespace, Resource, reqparse
from werkzeug.datastructures import FileStorage

from ..models import AnnotationModel, ImageModel, CategoryModel

api = Namespace('annotation', description='Annotation related operations')

create_update_delete = reqparse.RequestParser()
create_update_delete.add_argument('group_list', type=list, location='json')
create_update_delete.add_argument('image_id', type=int, location='json')

annotation_upload = reqparse.RequestParser()
annotation_upload.add_argument('xml_file', location='files', type=FileStorage, required=True, help='xml')
annotation_upload.add_argument('file_name', type=str, required=True)
annotation_upload.add_argument('file_type', type=str, required=True)


@api.route('/')
class Annotation(Resource):
    @login_required
    def get(self):
        """
        获取单个annotation对象
        :return:
        """
        return jsonify({'result': ' Success'})

    @login_required
    def delete(self):
        """
        删除某个标注
        :return:
        """
        annotation_id = request.args.get('annotation_id', '')
        if not annotation_id:
            msg = {'result': 'The annotation_id is necessary'}
            raise Exception(msg)
        annotation = AnnotationModel.objects.filter(id=annotation_id).first()
        if not annotation:
            msg = 'annotation_id {id}'.format(id=annotation_id)
            raise Exception(msg)
        annotation.delete()

        return {'result': 'Delete annotation success'}


@api.route('/all')
class AllAnnotation(Resource):
    @login_required
    def get(self):
        """
        获得某个图片的所有标注
        :return:
        """
        image_id = request.args.get('image_id', '')
        if not image_id:
            msg = {'result': 'The image_id is necessary'}
            raise Exception(msg)
        annotation_list = AnnotationModel.objects.filter(image_id=image_id).order_by('category_id',
                                                                                     'create_date')
        annotation_count = annotation_list.count()

        ade_list, agc_list, agc_fn_list, asc_h_list, asc_us_list, atr_list = [], [], [], [], [], []
        ec_list, emc_list = [], []
        hsil_list, lsil_list = [], []
        mic_list, met_list = [], []
        normal_list = []
        others_list = []
        scc_list, str_list = [], []
        yy_list = []
        no_label_list = []
        for annotation in annotation_list:
            if annotation.category_name == 'ade':
                ade_list.append(annotation)
            elif annotation.category_name == 'agc':
                agc_list.append(annotation)
            elif annotation.category_name == 'agc-fn':
                agc_fn_list.append(annotation)
            elif annotation.category_name == 'asc-h':
                asc_h_list.append(annotation)
            elif annotation.category_name == 'asc-us':
                asc_us_list.append(annotation)
            elif annotation.category_name == 'atr':
                atr_list.append(annotation)
            elif annotation.category_name == 'ec':
                ec_list.append(annotation)
            elif annotation.category_name == 'emc':
                emc_list.append(annotation)
            elif annotation.category_name == 'hsil':
                hsil_list.append(annotation)
            elif annotation.category_name == 'lsil':
                lsil_list.append(annotation)
            elif annotation.category_name == 'mic':
                mic_list.append(annotation)
            elif annotation.category_name == 'met':
                met_list.append(annotation)
            elif annotation.category_name == 'normal':
                normal_list.append(annotation)
            elif annotation.category_name == 'others':
                others_list.append(annotation)
            elif annotation.category_name == 'scc':
                scc_list.append(annotation)
            elif annotation.category_name == 'str':
                str_list.append(annotation)
            elif annotation.category_name == 'yy':
                yy_list.append(annotation)
            else:
                no_label_list.append(annotation)
        all_annotation = {
            'ade_list': ade_list,
            'agc_list': agc_list,
            'agc_fn_list': agc_fn_list,
            'asc_h_list': asc_h_list,
            'asc_us_list': asc_us_list,
            'atr_list': atr_list,
            'ec_list': ec_list,
            'emc_list': emc_list,
            'hsil_list': hsil_list,
            'lsil_list': lsil_list,
            'mic_list': mic_list,
            'met_list': met_list,
            'normal_list': normal_list,
            'others_list': others_list,
            'scc_list': scc_list,
            'str_list': str_list,
            'yy_list': yy_list,
            'no_label_list': no_label_list
        }
        return jsonify({'result': ' Success', 'annotation_list': all_annotation, 'annotation_count': annotation_count})

    @api.expect(create_update_delete)
    @login_required
    def post(self):
        """
        批量处理创建、更新、删除标注
        :return:
        """
        args = create_update_delete.parse_args()
        image_id = args.get('image_id')
        image = ImageModel.objects.filter(id=image_id).first()
        if not image:
            msg = {'result': 'This image is not exist  ——from create_annotation'}
            raise Exception(msg)

        group_list = args.get('group_list')
        for group in group_list:
            annotation_id = group['annotation_id']
            category_name = group['category_name']
            points = group['points']
            # 创建
            if group['status'] == 'create':
                key_point = points[0]
                x_list, y_list = [], []
                for point in points:
                    x_list.append(point['x'])
                    y_list.append(point['y'])
                x_min, x_max, y_min, y_max = min(x_list), max(x_list), min(y_list), max(y_list)
                bbox = [x_min, y_min, x_max, y_max]
                annotation = AnnotationModel(
                    image_id=image_id,
                    category_name=category_name,
                    coordinate_list=points,
                    key_point=key_point,
                    bbox=bbox
                )
                annotation.save()

            if group['status'] in ('update', 'delete'):
                annotation = AnnotationModel.objects.filter(id=annotation_id).first()
                if not annotation:
                    msg = {'result': 'The annotation_id is wrong  ——from update_or_delete_annotation'}
                    raise Exception(msg)
                if group['status'] == 'update':
                    annotation.category_name = category_name
                    annotation.save()
                if group['status'] == 'delete':
                    annotation.delete()
        return 'Save success!'

    @login_required
    def delete(self):
        """
        删除一张图的所有标注
        :return:
        """
        image_id = request.args.get('image_id', '')
        if image_id:
            image = ImageModel.objects.filter(id=image_id).first()
            if not image:
                msg = 'The image_id of params is not correct'
                raise Exception(msg)
        annotation_set = AnnotationModel.objects.filter(image_id=image_id)
        for annotation in annotation_set:
            annotation.delete()
        return 'Delete all success'


@api.route('/image/<int:image_id>')
class AnnotationUpload(Resource):
    @api.expect(annotation_upload)
    @login_required
    def post(self, image_id):
        """
        上传json文件给指定image添加标注
        :param annotation_id:
        :return:
        """
        image = ImageModel.objects.filter(id=image_id).first()
        if not image:
            msg = {'result': 'No such image'}
            raise Exception(msg)
        image_name = image.file_name

        args = annotation_upload.parse_args()
        # 设定标注颜色
        file_type = args.get('file_type')
        if file_type == 'manual':
            stroke_color = '#ffff00'
        else:
            stroke_color = '#ff4500'

        # 解析xml标注文件
        annotation_xml_file = args.get('xml_file', '')
        if not annotation_xml_file:
            msg = 'No annotation_xml_file'
            raise Exception(msg)
        full_name = args.get('file_name', '')
        file_name = os.path.splitext(full_name)[0]

        if file_name != image_name:
            return 'The xml_file is not belong to this image'

        root = ET.fromstring(annotation_xml_file.stream.read())
        object_list = root.findall('object')
        for object in object_list:
            # 分类ID及名称
            category_name = object.find('name').text
            category = CategoryModel.objects.filter(name=category_name).first()
            if not category:
                msg = 'No such category {category_name}'.format(category_name=category_name)
                raise Exception(msg)

            # 标注坐标
            point_list = object.find('segmentation').findall('points')
            coordinate_list = []
            for point in point_list:
                coordinate = {'x': float(point.find('x').text), 'y': float(point.find('y').text)}
                coordinate_list.append(coordinate)
            key_point = coordinate_list[0]

            # bbox
            bndbox = object.find('bndbox')
            x_min = int(bndbox.find('xmin').text)
            y_min = int(bndbox.find('ymin').text)
            x_max = int(bndbox.find('xmax').text)
            y_max = int(bndbox.find('ymax').text)
            bbox = [x_min, y_min, x_max, y_max]

            # 保存标注
            annotation = AnnotationModel(
                image_id=image.id,
                category_id=category.id,
                category_name=category_name,
                coordinate_list=coordinate_list,
                stroke_color=stroke_color,
                key_point=key_point,
                bbox=bbox
            )
            annotation.save()

        return '上传成功！'
