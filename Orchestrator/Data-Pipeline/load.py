from dotenv import load_dotenv

load_dotenv()

import os
import sys
# from llama_index.readers.mongo import SimpleMongoReader
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
# from llama_index.indices.vector_store.base import VectorStoreIndex
# from llama_index.storage.storage_context import StorageContext
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex


query_dict = {}

def load_index_mongodb():
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) # This is your Project Root
    DATA_PATH = os.path.join(ROOT_DIR, 'data')  # requires `import os`
    documents = SimpleDirectoryReader(DATA_PATH).load_data()


    # Create a new client and connect to the server
    client = MongoClient(os.getenv("MONGODB_URI"), server_api=ServerApi('1'))

    # create Atlas as a vector store
    store = MongoDBAtlasVectorSearch(
        client,
        db_name=os.getenv('MONGODB_DATABASE'),
        collection_name=os.getenv('MONGODB_VECTORS'), # this is where your embeddings will be stored
        index_name=os.getenv('MONGODB_VECTOR_INDEX') # this is the name of the index you will need to create
    )

    # now create an index from all the Documents and store them in Atlas
    storage_context = StorageContext.from_defaults(vector_store=store)


    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context,
        show_progress=True, # this will show you a progress bar as the embeddings are created
    )

def load_index(engineType):
    if(engineType == "mongodb"):
        load_index_mongodb()
    return True



sys.modules[__name__] = load_index