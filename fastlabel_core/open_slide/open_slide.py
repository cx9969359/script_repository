from flask_login import login_required
from flask_restplus import Namespace, Resource
from openslide import OpenSlide, deepzoom

from fastlabel_core import ImageModel

api = Namespace('slide', description='')
Format = 'jpeg'
Tile_Size = 256
Overlap = 1


@api.route('/dzi/<int:image_id>')
class Dzi(Resource):
    # @login_required
    def get(self, image_id):
        image = ImageModel.objects.filter(id=image_id).first()
        if not image:
            return {"message": "Invalid image id"}, 400
        image_path = image.path
        slide = OpenSlide(image_path)
        deep_zoom = deepzoom.DeepZoomGenerator(slide, tile_size=Tile_Size, overlap=Overlap)
        dzi = deep_zoom.get_dzi(format='jpeg')
        return dzi


@api.route('')
class TileFile(Resource):
    def get(self):
