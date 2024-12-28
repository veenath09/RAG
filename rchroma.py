from langchain.chains import RetrievalQA
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
import os

# Set Google API key
from dotenv import dotenv_values

dotenv_values()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Step 1: Setup Generative AI LLM
llm = GoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.environ["GOOGLE_API_KEY"]
)

# Step 2: Connect to a ChromaDB server
client = chromadb.HttpClient(host="127.0.0.1", port=8000)
  # Adjust port if needed

# Step 3: Create embeddings using the specified model
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Step 4: Create a retriever for fetching relevant documents
db = Chroma(client=client, embedding_function=embeddings)
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 1})

# Step 5 (Optional): Fetch and display similar documents
def pretty_print_docs(docs):
    """Pretty print retrieved documents with separators."""
    for i, doc in enumerate(docs, start=1):
        print(f"\n{'-' * 100}")
        print(f"Document {i}:\n{doc.page_content}\n")
        print(f"Metadata: {doc.metadata}")

# Retrieve documents similar to the query
query = "Tell us what is RAG Fusion"
docs = retriever.invoke(query)

# Display retrieved documents and their metadata
pretty_print_docs(docs)

# Step 6: Create a retrieval chain for querying the LLM
chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# Query the chain and get a response
response = chain.invoke("Tell us which module is most relevant to RAG Fusion type is under 50 words")

# Display the response and sources in a tidy format
print("\n" + "=" * 100)
print("Response:\n")
print(response["result"])
print("\nSources:\n")
for i, source in enumerate(response["source_documents"], start=1):
    print(f"Source {i}:\n{source.page_content}\nMetadata: {source.metadata}")
    print("=" * 100)
