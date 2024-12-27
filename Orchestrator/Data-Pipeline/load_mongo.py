## This script loads data from a mongo database into an index
## This will convert all the documents in the database into vectors
## which requires a call to OpenAI for each one, so it can take some time.
## Once the data is indexed, it will be stored as a new collection in mongodb
## and you can query it without having to re-index every time.
from dotenv import load_dotenv
import sys
import threading


load_dotenv()

# This will turn on really noisy logging if you want it, but it will slow things down
# import logging
# import sys
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

import os
# from llama_index.readers.mongo import SimpleMongoReader
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
# from llama_index.indices.vector_store.base import VectorStoreIndex
# from llama_index.storage.storage_context import StorageContext
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core import Settings
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

llm = AzureOpenAI(
    model=os.getenv("AZURE_LLM_MODEL"),
    deployment_name=os.getenv("AZURE_LLM_MODEL_DEPLOYMENT"),
    api_key=os.getenv("AZURE_API_KEY"),
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    api_version=os.getenv("AZURE_API_VERSION"),
)

# You need to deploy your own embedding model as well as your own chat completion model
embed_model = AzureOpenAIEmbedding(
    model=os.getenv("AZURE_LLM_MODEL_EMBED"),
    deployment_name=os.getenv("AZURE_LLM_MODEL_DEPLOYMENT"),
    api_key=os.getenv("AZURE_API_KEY_EMBED"),
    azure_endpoint=os.getenv("AZURE_ENDPOINT_EMBED"),
    api_version=os.getenv("AZURE_ENDPOINT_VERSION_EMBED"),
)

Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 512
Settings.chunk_overlap = 50

# load objects from mongo and convert them into LlamaIndex Document objects
# llamaindex has a special class that does this for you
# it pulls every object in a given collection

def load_index(path):

    ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) # This is your Project Root
    DATA_PATH = os.path.join(ROOT_DIR, 'data')  # requires `import os`
    DATA_PATH_IND = os.path.join(DATA_PATH, path)  # requires `import os`

    
    documents = SimpleDirectoryReader(DATA_PATH_IND).load_data()


    # Create a new client and connect to the server
    client = MongoClient(os.getenv("MONGODB_URI"), server_api=ServerApi('1'))

    # create Atlas as a vector store
    store = MongoDBAtlasVectorSearch(
        client,
        db_name=os.getenv('MONGODB_DATABASE'),
        collection_name=os.getenv('MONGODB_VECTORS'), # this is where your embeddings will be stored
        vector_index_name=os.getenv('MONGODB_VECTOR_INDEX') # this is the name of the index you will need to create
    )


    # now create an index from all the Documents and store them in Atlas
    storage_context = StorageContext.from_defaults(vector_store=store)



    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context,
        show_progress=True, # this will show you a progress bar as the embeddings are created
    )



t1 = threading.Thread(target=load_index, args=("2018",))
t2 = threading.Thread(target=load_index, args=("2019",))
t3 = threading.Thread(target=load_index, args=("2020",))
t4 = threading.Thread(target=load_index, args=("2021",))
t5 = threading.Thread(target=load_index, args=("2022",))
t6 = threading.Thread(target=load_index, args=("2023",))
t7 = threading.Thread(target=load_index, args=("2024",))

t1.start()
t2.start()
t3.start()
t4.start()
t5.start()
t6.start()
t7.start()

t1.join()
t2.join()
t3.join()
t4.join()
t5.join()
t6.join()
t7.join()

sys.modules[__name__] = load_index

