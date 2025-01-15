from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_chroma import Chroma
from pprint import pprint
import chromadb
import os 
from dotenv import load_dotenv

load_dotenv()


os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

llm = GoogleGenerativeAI(
    model = "gemini-1.5-flash",google_api_key=os.environ["GOOGLE_API_KEY"])

pdfloader = PyPDFDirectoryLoader("./pdf-docs")
documents = pdfloader.load()

textsplitter = RecursiveCharacterTextSplitter(chunk_size=2500,chunk_overlap=1000,separators=["."])


metadata_field_info = []

alldocuments = textsplitter.split_documents(documents)

""" for document in alldocuments:
    print(document) """
client = chromadb.HttpClient(host="127.0.0.1", port=8000) 

db = Chroma(client=client, embedding_function=embeddings)

retv = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})


# This example only specifies a relevant query
retdata = retv.invoke("what are documents contains about retreaval")

pprint(retdata)

print("test")