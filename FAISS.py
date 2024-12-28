
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
import os 
from dotenv import dotenv_values

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")


pdf_loader = PyPDFDirectoryLoader("./pdf-docs" )
pages_dir = pdf_loader.load()
#print(len(pages_dir))



loaders = [pdf_loader]
documents = []
for loader in loaders:
    documents.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
all_documents = text_splitter.split_documents(documents)

print(f"Total number of documents: {len(all_documents)}")

#Step 1 - setup OCI Generative AI llm


embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


#Step 2 - index documents and persist

# Set the batch size
batch_size = 96

# Calculate the number of batches
num_batches = len(all_documents) // batch_size + (len(all_documents) % batch_size > 0)


texts = ["FAISS is an important library", "LangChain supports FAISS"]
db = FAISS.from_texts(texts, embeddings)
retv = db.as_retriever()

# Iterate over batches
for batch_num in range(num_batches):
    # Calculate start and end indices for the current batch
    start_index = batch_num * batch_size
    end_index = (batch_num + 1) * batch_size
    # Extract documents for the current batch
    batch_documents = all_documents[start_index:end_index]
    # Your code to process each document goes here
    print(batch_documents)
    print("\n\n")
    #retv.add_documents(batch_documents)
    print(start_index, end_index)


db.save_local("faiss_index")









