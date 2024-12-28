from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
import os 
from pprint import pprint
import chromadb

from dotenv import dotenv_values

dotenv_values()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

llm = GoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.environ["GOOGLE_API_KEY"]
)

#setup client connection with chromadb

client = chromadb.HttpClient(host="127.0.0.1", port=8000)

#create an retrever 

db = Chroma(client=client, embedding_function=embeddings)

retv = db.as_retriever(search_type="similarity",search_kwargs={"k":4})


chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retv,
    return_source_documents = True
)

response =chain.invoke("What is RAG Fusion")


print("\n" + "=" * 100)
print("Response:\n")
print(response["result"])
print("\nSources:\n")
for i, source in enumerate(response["source_documents"], start=1):
    print(f"Source {i}:\n{source.page_content}\nMetadata: {source.metadata}")
    print("=" * 100)


