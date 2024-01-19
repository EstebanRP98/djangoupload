from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from decouple import config


class MongoDBConnector:
    def __init__(self):
        uri = config('DATABASE_URL')
        self.client = MongoClient(uri, server_api=ServerApi('1'))
        self.db = self.client['invoices-management']

    def get_collection(self, collection_name):
        return self.db[collection_name]
