from flask import Blueprint
from flask_restplus import Api

from .open_slide import api as ns_open_slide

from ..config import Config

# Create /api/ space
blueprint = Blueprint('', __name__,url_prefix='')

api = Api(
    blueprint,
    title=Config.NAME,
    version=Config.VERSION,
)

# Remove default namespace
api.namespaces.pop(0)

# Setup API namespaces
api.add_namespace(ns_open_slide)

