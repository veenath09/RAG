from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_chroma import Chroma
import os 

from dotenv import dotenv_values

dotenv_values()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")


#This demo creates Chroma Vector Store

# Step 1 - load and split documents

pdf_loader = PyPDFDirectoryLoader("./pdf-docs" )
loaders = [pdf_loader]

documents = []
for loader in loaders:
    documents.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=100)
all_documents = text_splitter.split_documents(documents)

print(f"Total number of documents: {type(all_documents)}")

for doc in all_documents:
    print(doc)
    

#Step 2 - setup OCI Generative AI llm



embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

#Step 3 - since OCIGenAIEmbeddings accepts only 96 documents in one run , we will input documents in batches.

# Set the batch size
batch_size = 96

# Calculate the number of batches
num_batches = len(all_documents) // batch_size + (len(all_documents) % batch_size > 0)

""" db  = Chroma(
    collection_name="mychromadb",
    embedding_function=embeddings,
    # other params...
)

db.add_documents(all_documents)
retv = db.as_retriever()

# Iterate over batches
for batch_num in range(num_batches):
    # Calculate start and end indices for the current batch
    start_index = batch_num * batch_size
    end_index = (batch_num + 1) * batch_size
    # Extract documents for the current batch
    batch_documents = all_documents[start_index:end_index]
    # Your code to process each document goes here
    retv.add_documents(batch_documents)
    print(start_index, end_index)

#Step 4 - here we persist the collection
#Since Chroma 0.4.x the manual persistence method is no longer supported as docs are automatically persisted.
#db.persist()
 """




