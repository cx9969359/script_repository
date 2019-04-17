import os

from ..models import ImageModel, CategoryModel, AnnotationModel


def scan_func(task, socket, dataset):
    rootDir = dataset.directory
    task.info(f"Scanning {rootDir}")
    walk_directory(rootDir, task, socket, dataset)


def walk_directory(rootDir, task, socket, dataset):
    list_dir = os.listdir(rootDir)
    count = 0
    for file_name in list_dir:
        path = os.path.join(rootDir, file_name)
        if os.path.isdir(path) and not ('_files' in file_name) and not ('thumbnail' in file_name):
            walk_directory(path, task, socket, dataset)
        else:
            set_progress_to_task(list_dir, path, task, socket)
            save_image_model_by_path(path, count, dataset, task)
    task.info(f"Created {count} new image(s)")
    task.set_progress(100, socket=socket)


def save_image_model_by_path(path, count, dataset, task):
    if path.endswith(ImageModel.PATTERN):
        db_image = ImageModel.objects(path=path).first()
        if db_image is not None:
            pass
        try:
            ImageModel.create_from_path(path, dataset.id).save()
            count += 1
            task.info(f"New file found: {path}")
        except:
            task.warning(f"Could not read {path}")


def set_progress_to_task(dir_list, directory, task, socket):
    try:
        youarehere = dir_list.index(directory.split('/')[-1])
        progress = int(((youarehere) / len(dir_list)) * 100)
        task.set_progress(progress, socket=socket)
    except:
        pass


def import_coco_func(task, socket, dataset, coco_json):
    task.info("Beginning Import")

    images = ImageModel.objects(dataset_id=dataset.id)
    categories = CategoryModel.objects

    coco_images = coco_json.get('images', [])
    coco_annotations = coco_json.get('annotations', [])
    coco_categories = coco_json.get('categories', [])

    task.info(f"Importing {len(coco_categories)} categories, "
              f"{len(coco_images)} images, and "
              f"{len(coco_annotations)} annotations")

    total_items = sum([
        len(coco_categories),
        len(coco_annotations),
        len(coco_images)
    ])
    progress = 0

    task.info("===== Importing Categories =====")
    # category id mapping  ( file : database )
    categories_id = {}

    # Create any missing categories
    for category in coco_categories:

        category_name = category.get('name')
        category_id = category.get('id')
        category_model = categories.filter(name__iexact=category_name).first()

        if category_model is None:
            task.warning(f"{category_name} category not found (creating a new one)")

            new_category = CategoryModel(
                name=category_name,
                keypoint_edges=category.get('skeleton', []),
                keypoint_labels=category.get('keypoints', [])
            )
            new_category.save()

            category_model = new_category
            dataset.categories.append(new_category.id)

        task.info(f"{category_name} category found")
        # map category ids
        categories_id[category_id] = category_model.id

        # update progress
        progress += 1
        task.set_progress((progress / total_items) * 100, socket=socket)

    dataset.update(set__categories=dataset.categories)

    task.info("===== Loading Images =====")
    # image id mapping ( file: database )
    images_id = {}

    # Find all images
    for image in coco_images:
        image_id = image.get('id')
        image_filename = image.get('file_name')

        # update progress
        progress += 1
        task.set_progress((progress / total_items) * 100, socket=socket)

        image_model = images.filter(file_name__exact=image_filename).all()

        if len(image_model) == 0:
            task.warning(f"Could not find image {image_filename}")
            continue

        if len(image_model) > 1:
            task.error(f"To many images found with the same file name: {image_filename}")
            continue

        task.info(f"Image {image_filename} found")
        image_model = image_model[0]
        images_id[image_id] = image_model

    task.info("===== Import Annotations =====")
    for annotation in coco_annotations:

        image_id = annotation.get('image_id')
        category_id = annotation.get('category_id')
        segmentation = annotation.get('segmentation', [])
        keypoints = annotation.get('keypoints', [])
        is_crowd = annotation.get('iscrowed', False)

        progress += 1
        task.set_progress((progress / total_items) * 100, socket=socket)

        if len(segmentation) == 0 and len(keypoints) == 0:
            task.warning(f"Annotation {annotation.get('id')} has no segmentation or keypoints")
            continue

        try:
            image_model = images_id[image_id]
            category_model_id = categories_id[category_id]
        except KeyError:
            task.warning(f"Could not find image assoicated with annotation {annotation.get('id')}")
            continue

        annotation_model = AnnotationModel.objects(
            image_id=image_model.id,
            category_id=category_model_id,
            segmentation=segmentation,
            keypoints=keypoints
        ).first()

        if annotation_model is None:
            task.info(f"Creating annotation data ({image_id}, {category_id})")

            annotation_model = AnnotationModel(image_id=image_model.id)
            annotation_model.category_id = category_model_id

            annotation_model.color = annotation.get('color')
            annotation_model.metadata = annotation.get('metadata', {})
            annotation_model.segmentation = segmentation
            annotation_model.keypoints = keypoints
            annotation_model.save()

            image_model.update(set__annotated=True)
        else:
            task.info(f"Annotation already exists (i:{image_id}, c:{category_id})")

    task.set_progress(100, socket=socket)
