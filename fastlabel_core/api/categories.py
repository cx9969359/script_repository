import datetime

from flask import jsonify
from flask_login import login_required, current_user
from flask_restplus import Namespace, Resource, reqparse
from mongoengine.errors import NotUniqueError

from ..models import CategoryModel
from ..util import query_util

api = Namespace('category', description='Category related operations')

create_category = reqparse.RequestParser()
create_category.add_argument('name', required=True, location='json')
create_category.add_argument('color', required=True, location='json')


@api.route('/')
class Category(Resource):

    @login_required
    def get(self):
        """
        返回所有的分类
        :return:
        """
        category_list = CategoryModel.objects.all().order_by('create_date')
        return jsonify({'result': 'Success', 'category_list': category_list})

    @api.expect(create_category)
    @login_required
    def post(self):
        """
        新建分类
        :return:
        """
        args = create_category.parse_args()
        name = args.get('name')
        color = args.get('color')
        if CategoryModel.objects.filter(name=name).count() > 0:
            return '该标签名已存在，请重新输入', 400
        try:
            category = CategoryModel(name=name, color=color)
            category.save()
        except NotUniqueError as e:
            return {'message': 'Category already exists. Check the undo tab to fully delete the category.'}, 400

        return '新建成功', 200


@api.route('/<int:category_id>')
class Category(Resource):

    @login_required
    def get(self, category_id):
        """ Returns a category by ID """
        category = current_user.categories.filter(id=category_id).first()

        if category is None:
            return {'success': False}, 400

        return query_util.fix_ids(category)

    @login_required
    def delete(self, category_id):
        """ Deletes a category by ID """
        category = current_user.categories.filter(id=category_id).first()
        if category is None:
            return {"message": "Invalid image id"}, 400

        category.update(set__deleted=True, set__deleted_date=datetime.datetime.now())
        return {'success': True}
