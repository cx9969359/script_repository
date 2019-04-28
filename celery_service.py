from __future__ import absolute_import

from celery import Celery
from celery_package.celery_config import Config

celery_app = Celery('fast_label')
celery_app.config_from_object(Config)
