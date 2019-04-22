from io import BytesIO

from openslide import OpenSlide


def get_image_thumbnail(image):
    """
    根据image_obj获取图片缩略图
    :param image:
    :return:
    """
    slide = OpenSlide(image.path)
    thumbnail = slide.get_thumbnail((250, 500))
    buffer = BytesIO()
    thumbnail.save(buffer, 'jpeg', qulity=95)
    thumbnail_bytes = buffer.getvalue()
    slide.close()
    return thumbnail_bytes
