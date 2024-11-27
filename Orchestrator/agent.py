
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.azure_openai import AzureOpenAI
from pydantic import Field
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core.llms import ChatMessage
from pydantic import BaseModel
from googlesearch import search
import requests
from bs4 import BeautifulSoup
import json
import urllib.parse
import faiss
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core import Settings
from llama_index.readers.json import JSONReader
from llama_index.core import SummaryIndex
from llama_index.readers.web import SimpleWebPageReader
from IPython.display import Markdown, display
from dotenv import load_dotenv
import os
import sys

import base64
from email.message import EmailMessage


from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import make_msgid

load_dotenv()
api_key = os.getenv('AZURE_API_KEY')
azure_endpoint = os.getenv('AZURE_ENDPOINT')
api_version = os.getenv('AZURE_API_VERSION')

SEARCH_RESULT_NUM=3
SCOPES = ["https://www.googleapis.com/auth/gmail.send"]

llm = AzureOpenAI(
    model="gpt-4-32k",
    deployment_name="gpt-4-32k",
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version,
)

# You need to deploy your own embedding model as well as your own chat completion model
embed_model = AzureOpenAIEmbedding(
    model="text-embedding-ada-002",
    deployment_name="text-embedding-ada-002",
    api_key=os.getenv('AZURE_API_KEY_EMBED'),
    azure_endpoint=os.getenv('AZURE_ENDPOINT_EMBED'),
    api_version=os.getenv('AZURE_ENDPOINT_VERSION_EMBED'),
)

Settings.llm = llm
Settings.embed_model = embed_model

chat_engine = SimpleChatEngine.from_defaults(llm=llm)

class Topic(BaseModel):
    topics: list

# All functions
def analyze_text(
    text: str = Field(
        description="input text"
    ),
) -> Topic:
    """a generated list of topic objects"""
    
    messages = [
    ChatMessage(
        role="system", content="generate 10 topics as JSON object only"
    ),
    ChatMessage(role="user", content=text),
    ]
    return llm.chat(messages)

def search_text(
    text: str = Field(
        description="input text"
    ),
):
    """searched contents"""
    
    res = search(text, num=1, stop=SEARCH_RESULT_NUM)
    for i in res:
        print(i)
        store_text(i)
    

def store_text(page):
    # dimensions of text-ada-embedding-002
    d = 1536
    faiss_index = faiss.IndexFlatL2(d)

    documents = SimpleWebPageReader(html_to_text=True).load_data(
    [page]
)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context
    )

    index.storage_context.persist()

def generate_text(
        text: str = Field(
        description="input text"
    )
    ) -> str:
    print("Task 2 started")
    # load index from disk
    vector_store = FaissVectorStore.from_persist_dir("./storage")
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store, persist_dir="./storage"
    )
    index = load_index_from_storage(storage_context=storage_context)

    query_engine = index.as_query_engine()
    answer = query_engine.query(text)

    print(answer.get_formatted_sources())
    print("query was:", text)
    print("answer was:", answer)
    # Simulate some work
    print("Task 2 finished")
    return answer

def expand_text(
        text: str = Field(
        description="input text"
    )
    ) -> str:
    print("Task 2 started")
    # load index from disk
    vector_store = FaissVectorStore.from_persist_dir("./storage")
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store, persist_dir="./storage"
    )
    index = load_index_from_storage(storage_context=storage_context)

    query_engine = index.as_query_engine(
                        system_prompt="expand on the content and generate a new text"
    
    )
    answer = query_engine.query(text)

    print(answer.get_formatted_sources())
    print("query was:", text)
    print("answer was:", answer)
    # Simulate some work
    print("Task 2 finished")
    return answer

def format_text(
    text: str = Field(
        description="input text"
    ),
):
    
    messages = [
    ChatMessage(
        role="system", content="generate nicely formated html from input"
    ),
    ChatMessage(role="user", content=text),
    ]
    return llm.chat(messages)

def gmail_send_message(
        text: str = Field(
        description="input text"
    ),
        email: str = Field(
        description="email"
    ),
):
  """Create and send an email message
  Print the returned  message id
  Returns: Message object, including message id

  Load pre-authorized user credentials from the environment.
  TODO(developer) - See https://developers.google.com/identity
  for guides on implementing OAuth2 for the application.
  """
  """Shows basic usage of the Gmail API.
  Lists the user's Gmail labels.
  """
  creds = None
  # The file token.json stores the user's access and refresh tokens, and is
  # created automatically when the authorization flow completes for the first
  # time.
  if os.path.exists("token.json"):
    creds = Credentials.from_authorized_user_file("token.json", SCOPES)
  # If there are no (valid) credentials available, let the user log in.
  if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
      creds.refresh(Request())
    else:
      flow = InstalledAppFlow.from_client_secrets_file(
          "credentials.json", SCOPES
      )
      creds = flow.run_local_server(port=0)
    # Save the credentials for the next run
    with open("token.json", "w") as token:
      token.write(creds.to_json())

#   creds, _ = google.auth.default()

  try:
    service = build("gmail", "v1", credentials=creds)
    message = EmailMessage()

    asparagus_cid = make_msgid()
    message.add_alternative(text)

    part = MIMEText(text, 'html')
    # message.set_content(text)

    message.attach(part)
    message["To"] = email
    message["From"] = "atnniya@gmail.com"
    message["Subject"] = "AI Daily News Letter "

    # encoded message
    encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()

    create_message = {"raw": encoded_message}
    # pylint: disable=E1101
    send_message = (
        service.users()
        .messages()
        .send(userId="me", body=create_message)
        .execute()
    )
    print(f'Message Id: {send_message["id"]}')
  except HttpError as error:
    print(f"An error occurred: {error}")
    send_message = None
  return send_message

def custom_handle_reasoning_failure() -> str:
    """response for error handling"""
    
    messages = [
    ChatMessage(
        role="system", content="generate the final response at least 500 words"
    ),
    ]
    return llm.chat(messages)

# Add all tools here 
analyze_tool = FunctionTool.from_defaults(analyze_text)
search_tool = FunctionTool.from_defaults(search_text)
generate_tool = FunctionTool.from_defaults(generate_text)
expand_tool = FunctionTool.from_defaults(expand_text)
format_tool = FunctionTool.from_defaults(format_text)
email_tool = FunctionTool.from_defaults(gmail_send_message)

# Adds all agents here 
agent = ReActAgent.from_tools([analyze_tool, search_tool, generate_tool, expand_tool, format_tool, email_tool ], llm=llm, verbose=True, max_iterations=15)



def chat():
   agent.chat("""Run following tasks on topic of new advancements with AIOps
           1 - Analyze the input text 
           2 - Search each topic 
           3 - Generate a text from the step 1 input and expand it
           4 - Format the content 
           5 - Send an email to amiryad.inventive@gmail.com via content of step 5  
           """)
   

sys.modules[__name__] = chat