import datetime
import json
import os
import time

import imantics as im
from PIL import Image
from flask_login import UserMixin, current_user
from flask_mongoengine import MongoEngine
from mongoengine.queryset.visitor import Q

from .config import Config

db = MongoEngine()


class DatasetModel(db.DynamicDocument):
    id = db.SequenceField(primary_key=True)
    name = db.StringField(required=True, unique=True)
    directory = db.StringField()

    creator = db.StringField(required=True)
    create_date = db.DateTimeField(default=datetime.datetime.now())
    deleted = db.BooleanField(default=False)
    deleted_date = db.DateTimeField()

    def save(self, *args, **kwargs):
        # 获取dataset路径
        directory = os.path.join(Config.DATASET_DIRECTORY, self.name + '/')
        if not os.path.exists(directory):
            os.makedirs(directory)

        self.directory = directory
        if current_user:
            self.creator = current_user.username
        else:
            self.creator = 'system'
        return super(DatasetModel, self).save(*args, **kwargs)

    def download_images(self, keywords, limit=100):

        task = TaskModel(
            name="Downloading {} images to {} with keywords {}".format(limit, self.name, keywords),
            dataset_id=self.id,
            group="Downloading Images"
        )

        def download_images(task, dataset, keywords, limit):
            def custom_print(string):
                __builtins__.print("%f -- %s" % (time.time(), string))

                print = dprint
                task.log()

            for keyword in args['keywords']:
                response = gid.googleimagesdownload()
                response.download({
                    "keywords": keyword,
                    "limit": args['limit'],
                    "output_directory": output_dir,
                    "no_numbering": True,
                    "format": "png",
                    "type": "photo",
                    "print_urls": True,
                    "print_paths": True,
                    "print_size": True
                })

        return task

    def import_coco(self, coco):
        from .util.task_util import import_coco_func
        task = TaskModel(
            name="Scanning {} for new images".format(self.name),
            dataset_id=self.id,
            group="Annotation Import"
        )
        task.save()
        task.start(import_coco_func, dataset=self, coco_json=coco)

        return task.api_json()

    def scan(self):
        from .util.task_util import scan_func
        task = TaskModel(
            name="Scanning {} for new images".format(self.name),
            dataset_id=self.id,
            group="Directory Image Scan"
        )
        task.save()
        task.start(scan_func, dataset=self)

        return task.api_json()

    def is_owner(self, user):

        if user.is_admin:
            return True

        return user.username.lower() == self.owner.lower()

    def can_download(self, user):
        return self.is_owner(user)

    def can_delete(self, user):
        return self.is_owner(user)

    def can_share(self, user):
        return self.is_owner(user)

    def can_generate(self, user):
        return self.is_owner(user)

    def can_edit(self, user):
        return self.is_owner(user)

    def permissions(self, user):
        return {
            'owner': self.is_owner(user),
            'edit': self.can_edit(user),
            'share': self.can_share(user),
            'generate': self.can_generate(user),
            'delete': self.can_delete(user),
            'download': self.can_download(user)
        }


class ImageModel(db.DynamicDocument):
    PATTERN = (".gif", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".dzi", ".wsi")
    _dataset = None

    id = db.SequenceField(primary_key=True)
    path = db.StringField(required=True, unique=True)
    prefix_path = db.StringField()

    dataset_id = db.IntField()

    width = db.IntField(required=True)
    height = db.IntField(required=True)
    file_name = db.StringField()
    file_type = db.StringField()
    # 切片图片的格式
    piece_format = db.StringField(default='')
    annotated = db.BooleanField(default=False)

    image_url = db.StringField()
    thumbnail_url = db.StringField(default='')
    create_date = db.DateTimeField(default=datetime.datetime.now())

    coco_url = db.StringField()

    deleted = db.BooleanField(default=False)

    @classmethod
    def create_from_path(cls, path, dataset_id=None):
        image = cls()
        if not os.path.isdir(os.path.join(path)):
            full_name = os.path.basename(path)
            target_len = len(path) - len(full_name)
            datasets_index = path.find('tif_images')
            image.prefix_path = path[datasets_index:target_len - 1] + '/'

            name_list = full_name.split('.')
            image.file_name = name_list[0]
            image.path = path

            if dataset_id is not None:
                image.dataset_id = dataset_id
            else:
                # Get dataset name from path
                folders = path.split('/')
                i = folders.index('datasets')
                dataset_name = folders[i + 1]
                dataset = DatasetModel.objects(name=dataset_name).first()
                if dataset is not None:
                    image.dataset_id = dataset.id

            pattern = name_list[-1]
            if pattern in ("gif", "png", "jpg", "jpeg", "bmp"):
                image.file_type = pattern
                pil_image = Image.open(path)
                image.width = pil_image.size[0]
                image.height = pil_image.size[1]
                pil_image.close()
                return image
            elif pattern == 'dzi':
                image.file_type = 'dzi'
                import xml.etree.cElementTree as ET
                root = ET.parse(path).getroot()
                image.piece_format = root.attrib['Format']
                for child in root:
                    if 'Size' in child.tag:
                        image.width = child.attrib['Width']
                        image.height = child.attrib['Height']
                return image
            elif pattern == 'tif':
                from openslide import OpenSlide
                image.file_type = 'tif'
                slide = OpenSlide(path)
                image.width = slide.dimensions[0]
                image.height = slide.dimensions[1]

                # 保存并获取缩略图路径
                thumbnail = slide.get_thumbnail((200, 500))
                thumbnail_name = '%s.jpeg' % image.file_name
                dir = os.path.join(Config.DATASET_DIRECTORY, '_thumbnail')
                if not os.path.exists(dir):
                    os.makedirs(dir)
                thumbnail_save_url = os.path.join(dir, thumbnail_name)
                thumbnail.save(thumbnail_save_url, 'jpeg')
                image.thumbnail_url = '/tif_images/_thumbnail/' + thumbnail_name
                slide.close()
                return image
            elif pattern == 'wsi':
                image.file_type = 'wsi'
                pass

    def thumbnail_path(self):
        folders = self.path.split('/')
        i = folders.index("datasets")
        folders.insert(i + 1, "_thumbnails")

        directory = '/'.join(folders[:-1])
        if not os.path.exists(directory):
            os.makedirs(directory)

        return '/'.join(folders)

    def thumbnail(self):
        image = self().draw(color_by_category=True, bbox=False)
        return Image.fromarray(image)

    def copy_annotations(self, annotations):
        """
        Creates a copy of the annotations for this image
        :param annotations: QuerySet of annotation models
        :return: number of annotations
        """
        annotations = annotations.filter(width=self.width, height=self.height, area__gt=0)

        for annotation in annotations:
            clone = annotation.clone()

            clone.dataset_id = self.dataset_id
            clone.image_id = self.id

            clone.save(copy=True)

        return annotations.count()

    @property
    def dataset(self):
        if self._dataset is None:
            self._dataset = DatasetModel.objects(id=self.dataset_id).first()
        return self._dataset

    def __call__(self):

        image = im.Image.from_path(self.path)
        for annotation in AnnotationModel.objects(image_id=self.id, deleted=False).all():
            if not annotation.is_empty():
                image.add(annotation())

        return image

    def can_delete(self, user):
        return user.can_delete(self.dataset)

    def can_download(self, user):
        return user.can_download(self.dataset)

    # TODO: Fix why using the functions throws an error
    def permissions(self, user):
        return {
            'delete': True,
            'download': True
        }


class AnnotationModel(db.DynamicDocument):
    id = db.SequenceField(primary_key=True)
    image_id = db.IntField(required=True)
    category_id = db.IntField(default=0)
    category_name = db.StringField(default='')

    coordinate_list = db.ListField(default=[])
    key_point = db.DictField(default={})
    stroke_width = db.FloatField(default=0.25)
    stroke_color = db.StringField(default='#000000')
    bbox = db.ListField(defualt=[])
    creator = db.StringField(required=True)
    create_date = db.DateTimeField(default=datetime.datetime.now)

    def save(self, *args, **kwargs):
        if current_user:
            self.creator = current_user.username
        else:
            self.creator = 'system'

        return super(AnnotationModel, self).save(*args, **kwargs)


class CategoryModel(db.DynamicDocument):
    id = db.SequenceField(primary_key=True)
    name = db.StringField(required=True)
    create_date = db.DateTimeField(default=datetime.datetime.now())


class LicenseModel(db.DynamicDocument):
    id = db.SequenceField(primary_key=True)
    name = db.StringField()
    url = db.StringField()


class TaskModel(db.DynamicDocument):
    id = db.SequenceField(primary_key=True)

    # Type of task: Importer, Exporter, Scanner, etc.
    group = db.StringField(required=True)
    name = db.StringField(required=True)
    desciption = db.StringField()

    creator = db.StringField()

    #: Start date of the executor 
    start_date = db.DateTimeField()
    #: End date of the executor 
    end_date = db.DateTimeField()
    completed = db.BooleanField(default=False)
    failed = db.BooleanField(default=False)

    # If any of the information is relevant to the task
    # it should be added
    dataset_id = db.IntField()
    image_id = db.IntField()
    category_id = db.IntField()

    progress = db.FloatField(default=0, min_value=0, max_value=100)

    logs = db.ListField(default=[])
    errors = db.IntField(default=0)
    warnings = db.IntField(default=0)

    priority = db.IntField()

    metadata = db.DictField(default={})

    _update_every = 10
    _progress_update = 0

    def error(self, string):
        self._log(string, level="ERROR")

    def warning(self, string):
        self._log(string, level="WARNING")

    def info(self, string):
        self._log(string, level="INFO")

    def _log(self, string, level):

        level = level.upper()
        date = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")

        message = f"[{date}] [{level}] {string}"

        statment = {
            'push__logs': message
        }

        if level == "ERROR":
            statment['inc__errors'] = 1
            self.errors += 1

        if level == "WARNING":
            statment['inc__warnings'] = 1
            self.warnings += 1

        self.update(**statment)

    def set_progress(self, percent, socket=None):

        self.update(progress=int(percent), completed=(percent >= 100))

        # Send socket update every 10%
        if self._progress_update < percent or percent >= 100:

            if socket is not None:
                socket.emit('taskProgress', {
                    'id': self.id,
                    'progress': percent,
                    'errors': self.errors,
                    'warnings': self.warnings
                }, broadcast=True)

            self._progress_update += self._update_every

    def start(self, target, *args, **kwargs):

        from .sockets import socketio

        thread = socketio.start_background_task(
            target,
            task=self,
            socket=socketio,
            *args,
            **kwargs
        )
        return thread

    def api_json(self):
        return {
            "id": self.id,
            "name": self.name
        }


class CocoImportModel(db.DynamicDocument):
    id = db.SequenceField(primary_key=True)
    creator = db.StringField(required=True)
    progress = db.FloatField(default=0.0, min_value=0.0, max_value=1.0)
    errors = db.ListField(default=[])


class UserModel(db.DynamicDocument, UserMixin):
    password = db.StringField(required=True)
    username = db.StringField(max_length=25, required=True, unique=True)
    email = db.StringField(max_length=30)

    name = db.StringField()
    last_seen = db.DateTimeField()

    is_admin = db.BooleanField(default=False)

    preferences = db.DictField(default={})
    permissions = db.ListField(defualt=[])

    def save(self, *args, **kwargs):

        self.last_seen = datetime.datetime.now()

        return super(UserModel, self).save(*args, **kwargs)

    @property
    def datasets(self):
        self._update_last_seen()

        if self.is_admin:
            return DatasetModel.objects

        return DatasetModel.objects(Q(owner=self.username) | Q(users__contains=self.username))

    @property
    def categories(self):
        self._update_last_seen()

        if self.is_admin:
            return CategoryModel.objects

        dataset_ids = self.datasets.distinct('categories')
        return CategoryModel.objects(Q(id__in=dataset_ids) | Q(creator=self.username))

    @property
    def images(self):
        self._update_last_seen()

        if self.is_admin:
            return ImageModel.objects

        dataset_ids = self.datasets.distinct('id')
        return ImageModel.objects(dataset_id__in=dataset_ids)

    @property
    def annotations(self):
        self._update_last_seen()

        if self.is_admin:
            return AnnotationModel.objects

        image_ids = self.images.distinct('id')
        return AnnotationModel.objects(image_id__in=image_ids)

    def can_view(self, model):
        if model is None:
            return False

        return model.can_view(self)

    def can_download(self, model):
        if model is None:
            return False

        return model.can_download(self)

    def can_delete(self, model):
        if model is None:
            return False
        return model.can_delete(self)

    def can_edit(self, model):
        if model is None:
            return False

        return model.can_edit(self)

    def _update_last_seen(self):
        self.update(last_seen=datetime.datetime.now())


# https://github.com/MongoEngine/mongoengine/issues/1171
# Use this methods until a solution is found
def upsert(model, query=None, update=None):
    if not update:
        update = query

    if not query:
        return None

    found = model.objects(**query)

    if found.first():
        return found.modify(new=True, **update)

    new_model = model(**update)
    new_model.save()

    return new_model


def create_from_json(json_file):
    with open(json_file) as file:

        data_json = json.load(file)
        for category in data_json.get('categories', []):
            name = category.get('name')
            if name is not None:
                upsert(CategoryModel, query={"name": name}, update=category)

        for dataset_json in data_json.get('datasets', []):
            name = dataset_json.get('name')
            if name:
                # map category names to ids; create as needed
                category_ids = []
                for category in dataset_json.get('categories', []):
                    category_obj = {"name": category}

                    category_model = upsert(CategoryModel, query=category_obj)
                    category_ids.append(category_model.id)

                dataset_json['categories'] = category_ids
                upsert(DatasetModel, query={"name": name}, update=dataset_json)
