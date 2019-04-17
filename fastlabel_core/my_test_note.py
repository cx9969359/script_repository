import datetime
from openslide import OpenSlide,deepzoom
import os


class A:
    def a(self):
        path ='E:\\open_tif\\have.tif'
        os_path = os.path.join(path)
        a = os.path.abspath(os.path.join(os_path,'../'))
        print(a)
        list_path = os.path.split(os_path)
        # slide = OpenSlide(path)
        # deep_zoom = deepzoom.DeepZoomGenerator(slide, tile_size=256,overlap=1)
        # print(deep_zoom)
        # thumbnail = slide.get_thumbnail((200,500))
        # print(thumbnail)
        # file_name = 'have'
        # thumbnail_name = '%s.jpeg' % file_name
        # print(thumbnail_name)
        # thumbnail.save('F:\\tif_images\\thyroid\\have.jpeg',format='jpeg')

        # dzi = deep_zoom.get_dzi(format='jpeg')
        # print(dzi)


if __name__ == '__main__':
    a = A()
    a.a()
