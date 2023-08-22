import openai
import os
from secret_key import open_ai_key, pinecone_api_key, pinecone_env
import pandas as pd
from langchain.document_loaders import CSVLoader
from langchain.chains import RetrievalQA
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI
from langchain.embeddings import SentenceTransformerEmbeddings

os.environ['OPENAI_API_KEY'] = open_ai_key
path = "data\\chatbot_ecommerce_cleaned_data.csv"
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

def csv_load(directory):
    loader = CSVLoader(directory, encoding="utf8")
    documents = loader.load()
    return documents


documents = csv_load(path)
print(documents)

# def csv_load(directory):
#     loader = CSVLoader(directory, encoding="utf8")
#     index_creator = VectorstoreIndexCreator()
#     docsearch = index_creator.from_loaders([loader])
#     return docsearch
 
# documents = csv_load(path)







