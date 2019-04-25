from flask import Flask, request
from flask_restful import Resource, Api

application = Flask(__name__)
api = Api(application)


class HelloWorld(Resource):
    def get(self):
        return {'about': 'Hello World!'}

    def post(self):
        some_json = request.get_json()
        return {'you sent' : some_json}, 201

class FetchGroups(Resource):
    def get(self):
        return {'groups': []}
api.add_resource(FetchGroups, '/groups')
api.add_resource(HelloWorld, '/')

if __name__ == '__main__':
    app.run(debug=True)
