from __future__ import absolute_import
from celery_package.celery_service import celery_app


@celery_app.task
def one(x, y):
    return x + y
