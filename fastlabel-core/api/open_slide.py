from openslide import OpenSlide, deepzoom
from flask_login import login_required
from flask_restplus import reqparse, Resource, Namespace

api = Namespace('slide', description='')
Format = 'jpeg'
Tile_Size = 256
Overlap = 1


@api.route('/')
class SlideFile():
    @login_required
    def get_tile(self):
        pass


@api.route('/dzi')
class Dzi():
    @login_required
    def get_dzi(self):
        slide = OpenSlide(path)
        deep_zoom = deepzoom.DeepZoomGenerator(slide, tile_size=Tile_Size, overlap=Overlap)
        image_dzi = deep_zoom.get_dzi(format=Format)
        return image_dzi
