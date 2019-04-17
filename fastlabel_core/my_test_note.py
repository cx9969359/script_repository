import datetime
from openslide import OpenSlide,deepzoom



class A:
    def a(self):
        slide = OpenSlide('E:\\open_tif\\have.tif')
        deep_zoom = deepzoom.DeepZoomGenerator(slide, tile_size=256,overlap=1)
        print(deep_zoom)
        dzi = deep_zoom.get_dzi(format='jpeg')
        print(dzi)


if __name__ == '__main__':
    a = A()
    a.a()
