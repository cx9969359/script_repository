from threading import Thread

from flask import request
from flask_login import login_required
from flask_restplus import Namespace, Resource, reqparse
from google_images_download import google_images_download as gid
from werkzeug.datastructures import FileStorage

from fastlabel_core.util.query_util import fix_ids
from ..models import *
from ..util import query_util, coco_util
from ..util.pagination_util import Pagination

api = Namespace('dataset', description='Dataset related operations')

dataset_create = reqparse.RequestParser()
dataset_create.add_argument('name', required=True)

page_data = reqparse.RequestParser()
page_data.add_argument('folder', default='', help='Folder for data')
page_data.add_argument('page', default=1, type=int)
page_data.add_argument('limit', default=20, type=int)

delete_data = reqparse.RequestParser()
delete_data.add_argument('fully', default=False, type=bool,
                         help="Fully delete dataset (no undo)")

coco_upload = reqparse.RequestParser()
coco_upload.add_argument('coco', location='files', type=FileStorage, required=True, help='Json coco')

update_dataset = reqparse.RequestParser()
update_dataset.add_argument('categories', location='json', type=list, help="New list of categories")
update_dataset.add_argument('default_annotation_metadata', location='json', type=dict,
                            help="Default annotation metadata")

dataset_generate = reqparse.RequestParser()
dataset_generate.add_argument('keywords', location='json', type=list, default=[],
                              help="Keywords associated with images")
dataset_generate.add_argument('limit', location='json', type=int, default=100, help="Number of images per keyword")

share = reqparse.RequestParser()
share.add_argument('users', location='json', type=list, default=[], help="List of users")

add_administrator = reqparse.RequestParser()
add_administrator.add_argument('username', location='json', type=str, required=True,
                               help='Add administrator to dataset')


@api.route('/')
class Dataset(Resource):
    @login_required
    def get(self):
        """ Returns all datasets """
        return query_util.fix_ids(current_user.datasets.filter(deleted=False).all())

    @api.expect(dataset_create)
    @login_required
    def post(self):
        """
        创建一个dataset
        :return:
        """
        name = dataset_create.parse_args().get('name', '')
        if not name:
            msg = {'result': 'Please entry dataset_name'}
            raise Exception(msg)
        if DatasetModel.objects.filter(name=name).count() > 0:
            msg = {'result': 'This datasetname has existed'}
            raise Exception(msg)
        dataset = DatasetModel(name=name)
        dataset.save()
        dataset_jsonObj = query_util.fix_ids(dataset)
        return {'result': 'success', 'dataset': dataset_jsonObj}


def download_images(output_dir, args):
    for keyword in args['keywords']:
        response = gid.googleimagesdownload()
        response.download({
            "keywords": keyword,
            "limit": args['limit'],
            "output_directory": output_dir,
            "no_numbering": True,
            "format": "jpg",
            "type": "photo",
            "print_urls": False,
            "print_paths": False,
            "print_size": False
        })


@api.route('/<int:dataset_id>/generate')
class DatasetGenerate(Resource):
    @api.expect(dataset_generate)
    @login_required
    def post(self, dataset_id):
        """ Adds images found on google to the dataset """
        args = dataset_generate.parse_args()

        dataset = current_user.datasets.filter(id=dataset_id, deleted=False).first()
        if dataset is None:
            return {"message": "Invalid dataset id"}, 400

        if not dataset.is_owner(current_user):
            return {"message": "You do not have permission to download the dataset's annotations"}, 403

        thread = Thread(target=download_images, args=(dataset.directory, args))
        thread.start()

        return {"success": True}


@api.route('/<int:dataset_id>')
class DatasetId(Resource):
    @login_required
    def delete(self, dataset_id):
        """ Deletes dataset by ID (only owners)"""

        dataset = DatasetModel.objects(id=dataset_id, deleted=False).first()

        if dataset is None:
            return {"message": "Invalid dataset id"}, 400

        if not current_user.can_delete(dataset):
            return {"message": "You do not have permission to delete the dataset"}, 403

        dataset.update(set__deleted=True, set__deleted_date=datetime.datetime.now())
        return {"success": True}

    @api.expect(update_dataset)
    def post(self, dataset_id):
        """ Updates dataset by ID """
        dataset = current_user.datasets.filter(id=dataset_id, deleted=False).first()
        if dataset is None:
            return {"message": "Invalid dataset id"}, 400

        args = update_dataset.parse_args()
        categories = args.get('categories')
        default_annotation_metadata = args.get('default_annotation_metadata')

        if categories is not None:
            dataset.categories = CategoryModel.bulk_create(categories)

        if default_annotation_metadata is not None:
            dataset.default_annotation_metadata = default_annotation_metadata

        dataset.update(
            categories=dataset.categories,
            default_annotation_metadata=dataset.default_annotation_metadata
        )

        return {"success": True}


@api.route('/<int:dataset_id>/share')
class DatasetIdShare(Resource):
    @api.expect(share)
    @login_required
    def post(self, dataset_id):
        args = share.parse_args()

        dataset = current_user.datasets.filter(id=dataset_id, deleted=False).first()
        if dataset is None:
            return {"message": "Invalid dataset id"}, 400

        if not dataset.is_owner(current_user):
            return {"message": "You do not have permission to share this dataset"}, 403

        dataset.update(users=args.get('users'))

        return {"success": True}


@api.route('/data')
class DatasetData(Resource):
    @login_required
    def get(self):
        """
        返回当前用户可编辑查看的所有data_set
        :return:
        """
        current_username = current_user.username
        # 超级管理员可查看编辑所有data_set
        if current_user.is_admin:
            datasets = DatasetModel.objects.all().order_by('create_date')
            datasets_json = []
            for dataset in datasets:
                dataset_json = query_util.fix_ids(dataset)
                images = ImageModel.objects(dataset_id=dataset.id)

                dataset_json['numberImages'] = images.count()
                dataset_json['numberAnnotated'] = images.filter(annotated=True).count()
                dataset_json['permissions'] = dataset.permissions(current_user)
                dataset_json['is_creator'] = 'true'

                first = images.first()
                if first is not None:
                    dataset_json['first_image_id'] = images.first().id
                    dataset_json['first_image_name'] = images.first().file_name
                    dataset_json['first_image_type'] = images.first().file_type
                datasets_json.append(dataset_json)
            return {"datasets": datasets_json}
        else:
            # 普通用户
            datasets = DatasetModel.objects(creator=current_username).order_by('create_date')

            datasets_json = []
            for dataset in datasets:
                dataset_json = query_util.fix_ids(dataset)
                images = ImageModel.objects(dataset_id=dataset.id)

                dataset_json['numberImages'] = images.count()
                dataset_json['numberAnnotated'] = images.filter(annotated=True).count()
                dataset_json['permissions'] = dataset.permissions(current_user)
                dataset_json['is_creator'] = 'true'

                first = images.first()
                if first is not None:
                    dataset_json['first_image_id'] = images.first().id
                    dataset_json['first_image_name'] = images.first().file_name
                    dataset_json['first_image_type'] = images.first().file_type
                datasets_json.append(dataset_json)

            datasets_created_by_others = DatasetModel.objects(creator__not__iexact=current_username).order_by(
                'create_date')
            for data_set in datasets_created_by_others:
                can_edit_username_list = [user['username'] for user in data_set.administrator_list]
                if current_username in can_edit_username_list:
                    dataset_json = query_util.fix_ids(data_set)
                    images = ImageModel.objects(dataset_id=data_set.id)

                    dataset_json['numberImages'] = images.count()
                    dataset_json['numberAnnotated'] = images.filter(annotated=True).count()
                    dataset_json['permissions'] = data_set.permissions(current_user)
                    dataset_json['is_creator'] = 'false'

                    first = images.first()
                    if first is not None:
                        dataset_json['first_image_id'] = images.first().id
                        dataset_json['first_image_name'] = images.first().file_name
                        dataset_json['first_image_type'] = images.first().file_type
                    datasets_json.append(dataset_json)

            return {"datasets": datasets_json}


@api.route('/<int:dataset_id>/data')
class DatasetDataId(Resource):

    @api.expect(page_data)
    @login_required
    def get(self, dataset_id):
        """ Endpoint called by image viewer client """

        exec_start = datetime.datetime.now()
        args = page_data.parse_args()
        limit = args['limit']
        page = args['page']
        folder = args['folder']

        args = dict(request.args)
        if args.get('limit') != None:
            del args['limit']
        if args.get('page') != None:
            del args['page']
        if args.get('folder') != None:
            del args['folder']

        query = {}
        for key, value in args.items():
            lower = value.lower()
            if lower in ["true", "false"]:
                value = json.loads(lower)

            if len(lower) != 0:
                query[key] = value

        # Check if dataset exists
        dataset = current_user.datasets.filter(id=dataset_id, deleted=False).first()
        if dataset is None:
            return {'message', 'Invalid dataset id'}, 400

        # Make sure folder starts with is in proper format
        if len(folder) > 0:
            folder = folder[0].strip('/') + folder[1:]
            if folder[-1] != '/':
                folder = folder + '/'

        # Get directory
        directory = os.path.join(dataset.directory, folder)
        if not os.path.exists(directory):
            return {'message': 'Directory does not exist.'}, 400

        images = ImageModel.objects(dataset_id=dataset_id, path__startswith=directory, **query).order_by('create_date')
        pagination = Pagination(images.count(), limit, page)
        images = images[pagination.start:pagination.end]

        images_json = []
        for image in images:
            image_json = query_util.fix_ids(image)

            annotation_set = AnnotationModel.objects(image_id=image.id)
            category_list = []
            for annotation in annotation_set:
                if annotation.category_name not in category_list:
                    category_list.append(annotation.category_name)
            categories = []
            for category_name in category_list:
                category = CategoryModel.objects.filter(category_name=category_name).first()
                obj = {}
                obj['name'] = category_name
                if not category:
                    obj['color'] = '#FF0000'
                else:
                    obj['color'] = category.color
                categories.append(obj)
            image_json['categories'] = categories
            image_json['annotations'] = annotation_set.count()
            image_json['permissions'] = image.permissions(current_user)

            images_json.append(image_json)

        subdirectories = [f for f in sorted(os.listdir(directory))
                          if os.path.isdir(directory + f)]

        delta = datetime.datetime.now() - exec_start
        return {
            "time_ms": int(delta.total_seconds() * 1000),
            "pagination": pagination.export(),
            "images": images_json,
            "folder": folder,
            "directory": directory,
            "dataset": query_util.fix_ids(dataset),
            "subdirectories": subdirectories
        }


@api.route('/<int:dataset_id>/coco')
class DatasetCoco(Resource):

    @login_required
    def get(self, dataset_id):
        """ Returns coco of images and annotations in the dataset (only owners) """
        dataset = current_user.datasets.filter(id=dataset_id).first()

        if dataset is None:
            return {"message": "Invalid dataset ID"}, 400

        if not current_user.can_download(dataset):
            return {"message": "You do not have permission to download the dataset's annotations"}, 403

        return coco_util.get_dataset_coco(dataset)

    @api.expect(coco_upload)
    @login_required
    def post(self, dataset_id):
        """ Adds coco formatted annotations to the dataset """
        args = coco_upload.parse_args()
        coco = args['coco']

        dataset = current_user.datasets.filter(id=dataset_id).first()
        if dataset is None:
            return {'message': 'Invalid dataset ID'}, 400

        return dataset.import_coco(json.load(coco))


@api.route('/administration/<int:dataset_id>')
class DataSetAdministration(Resource):

    @login_required
    def get(self, dataset_id):
        dataset = DatasetModel.objects(id=dataset_id).first()
        if not dataset:
            return {'message': 'Invalid dataset ID'}, 400
        return fix_ids(dataset)

    @api.expect(add_administrator)
    @login_required
    def post(self, dataset_id):
        dataset = DatasetModel.objects(id=dataset_id).first()
        if not dataset:
            return {'message': 'Invalid dataset ID'}, 400
        username = add_administrator.parse_args().get('username')
        user = UserModel.objects(username=username).first()
        print(user)
        user_name_list = [user['username'] for user in dataset.administrator_list]
        if (username not in user_name_list) and (username != dataset.creator):
            obj = {}
            obj['username'] = username
            obj['add_time'] = datetime.datetime.now().strftime('%Y-%m-%d')
            dataset.administrator_list.append(obj)
            dataset.save()
        return 'Add success'

    @login_required
    def delete(self, dataset_id):
        dataset = DatasetModel.objects(id=dataset_id).first()
        if not dataset:
            return {'message': 'Invalid dataset ID'}, 400
        username = request.args.get('username')
        user_username_list = [user['username'] for user in dataset.administrator_list]
        if username not in user_username_list:
            return {'message': 'No such an administrator of this dataset'}, 400
        else:
            for index, item in enumerate(dataset.administrator_list):
                if item['username'] == username:
                    dataset.administrator_list.pop(index)
                    dataset.save()
                    break
            return 'Remove success'


@api.route('/<int:dataset_id>/scan')
class DatasetScan(Resource):

    @login_required
    def get(self, dataset_id):
        dataset = DatasetModel.objects(id=dataset_id).first()

        if not dataset:
            return {'message': 'Invalid dataset ID'}, 400

        return dataset.scan()
