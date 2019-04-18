from io import BytesIO

from flask import make_response
from flask_restplus import Namespace, Resource
from openslide import OpenSlide, deepzoom

from fastlabel_core import ImageModel

api = Namespace('open-slide', description='')
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


@api.route('/<path:path>_files/<int:level>/<int:col>_<int:row>.<format>')
class TileFile(Resource):
    def get(self, path, level, col, row, format):
        slide = OpenSlide(path)
        deep_zoom = deepzoom.DeepZoomGenerator(slide, tile_size=Tile_Size, overlap=Overlap)
        format = format.lower()
        tile = deep_zoom.get_tile(level, (col, row))
        buffer = BytesIO()
        # quality范围1-95，默认75
        tile.save(buffer, format, qulity=90)
        tile_bytes = buffer.getvalue()
        res = make_response(tile_bytes)
        res.mimetype = 'image/ %s' % format
        slide.close()
        return res
