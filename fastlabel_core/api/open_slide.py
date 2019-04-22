import os
from io import BytesIO
from flask_login import login_required

from flask import make_response
from flask_restplus import Namespace, Resource
from openslide import OpenSlide, deepzoom

from fastlabel_core import ImageModel
from fastlabel_core.api.service.open_slidea_service import get_image_thumbnail

api = Namespace('open-slide', description='')
Format = 'jpeg'
Tile_Size = 256
Overlap = 1


@api.route('/thumbnail/<int:image_id>')
class Thumbnail(Resource):
    @login_required
    def get(self, image_id):
        image = ImageModel.objects.filter(id=image_id).first()
        if not image:
            return {"message": "Invalid image id"}, 400
        thumbnail_bytes = get_image_thumbnail(image)
        res = make_response(thumbnail_bytes)
        res.mimetype = 'image/ %s' % 'jpeg'
        return res


@api.route('/<path:path>_files/<int:level>/<int:col>_<int:row>.<format>')
class TileFile(Resource):
    def get(self, path, level, col, row, format):
        path = '/' + path
        print(os.path.join(path))
        slide = OpenSlide(os.path.join(path))
        deep_zoom = deepzoom.DeepZoomGenerator(slide, tile_size=Tile_Size, overlap=Overlap)
        format = format.lower()
        tile = deep_zoom.get_tile(level, (col, row))
        buffer = BytesIO()
        # quality范围1-95，默认75
        tile.save(buffer, format, qulity=95)
        tile_bytes = buffer.getvalue()
        res = make_response(tile_bytes)
        res.mimetype = 'image/ %s' % format
        slide.close()
        return res
