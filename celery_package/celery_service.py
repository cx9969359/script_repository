from __future__ import absolute_import

from celery import Celery


celery_app = Celery('kil', backend='redis://127.0.0.1:6379/0', broker='redis://127.0.0.1:6379/0')
