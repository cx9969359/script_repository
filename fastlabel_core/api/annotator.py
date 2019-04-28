from flask.json import jsonify
from flask_restplus import Namespace, Resource
from flask_login import login_required
from flask import request

from ..util import coco_util
from ..models import *

api = Namespace('annotator', description='Annotator related operations')


@api.route('/data')
class AnnotatorData(Resource):

    @login_required
    def post(self):
        """
        Called when saving data from the annotator client
        """
        data = request.get_json(force=True)
        image = data.get('image')
        image_id = image.get('id')

        image_model = ImageModel.objects(id=image_id).first()

        if image_model is None:
            return {'success': False, 'message': 'Image does not exist'}, 400

        # Check if current user can access dataset
        if current_user.datasets.filter(id=image_model.dataset_id).first() is None:
            return {'success': False, 'message': 'Could not find associated dataset'}

        categories = CategoryModel.objects.all()
        annotations = AnnotationModel.objects(image_id=image_id)

        current_user.update(preferences=data.get('user', {}))

        annotated = False
        # Iterate every category passed in the data
        for category in data.get('categories', []):
            # Find corresponding category object in the database
            db_category = categories.filter(id=category['id']).first()
            if db_category is None:
                continue

            db_category.update(
                set__color=category.get('color'),
                set__keypoint_edges=category.get('keypoint_edges', []),
                set__keypoint_labels=category.get('keypoint_labels', [])
            )

            # Iterate every annotation from the data annotations
            for annotation in category.get('annotations', []):

                # Find corresponding annotation object in database
                annotation_id = annotation.get('id')
                db_annotation = annotations.filter(id=annotation_id).first()

                if db_annotation is None:
                    continue

                # Paperjs objects are complex, so they will not always be passed. Therefor we update
                # the annotation twice, checking if the paperjs exists.

                # Update annotation in database
                db_annotation.update(
                    set__keypoints=annotation.get('keypoints', []),
                    set__metadata=annotation.get('metadata'),
                    set__color=annotation.get('color')
                )

                paperjs_object = annotation.get('compoundPath', [])

                # Update paperjs if it exists
                if len(paperjs_object) == 2:

                    width = db_annotation.width
                    height = db_annotation.height

                    # Generate coco formatted segmentation data
                    segmentation, area, bbox = coco_util. \
                        paperjs_to_coco(width, height, paperjs_object)

                    db_annotation.update(
                        set__segmentation=segmentation,
                        set__area=area,
                        set__bbox=bbox,
                        set__paper_object=paperjs_object,
                    )

                    if area > 0:
                        annotated = True

        image_model.update(
            set__metadata=image.get('metadata', {}),
            set__annotated=annotated,
            set__category_ids=image.get('category_ids', [])
        )

        return data


@api.route('/data')
class AnnotatorData(Resource):

    @login_required
    def post(self):
        data = request.get_json(force=True)
        image = data.get('image')
        image_id = image.get('id')
        category = data.get('category')
        category_id = category.get('id')
        category_name = category.get('name')
        dataset_id = data.get('dataset_id')
        coordinate = data.get('coordinate')
        bbox = data.get('bbox')

    def get(self):
        from celery_package.tasks import one
        r = one.apply_async()
        while True:
            if r.status == 'SUCCESS':
                return jsonify({'status': r.status, 'result': r.result})
