from __future__ import absolute_import

import json

from celery_service import celery_app


@celery_app.task
def one():
    return json.dumps('x')
