from fastlabel_core import ImageModel, AnnotationModel
from fastlabel_core.config import Config


def handle_result_dict(file_path, dict):
    """
    处理算法计算的结果
    :param file_path:
    :param dict:
    :return:
    """
    file_name = file_path.split('.')[0]
    image = ImageModel.objects.filter(file_name=file_name).first()
    if not image:
        return 'No such image when handle arithmetic result!', 400
    image_id = image.id
    for category, bbox_list in dict.items():
        for bbox in bbox_list:
            x_min, y_min, x_max, y_max = bbox[0], bbox[1], bbox[2], bbox[3]
            coordinate_list = [{'x': x_min, 'y': y_min}, {'x': x_min, 'y': y_max}, {'x': x_max, 'y': y_min},
                               {'x': x_max, 'y': y_max}]
            annotation = AnnotationModel(
                image_id=image_id,
                category_name=category,
                coordinate_list=coordinate_list,
                key_point=coordinate_list[1],
                bbox=bbox,
                stroke_color=Config.COMPUTER_STROKE_COLOR,
            )
            annotation.save()
