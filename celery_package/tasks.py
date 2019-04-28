from __future__ import absolute_import

from flask.json import jsonify

from celery_service import celery_app
from fastlabel_core import ImageModel
import json

@celery_app.task
def one():
    return json.dumps('x')
