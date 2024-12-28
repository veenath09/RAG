from langchain.chains import RetrievalQA
import chromadb
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings,GoogleGenerativeAI
import os

#In this demo we will retrieve documents and send these as a context to the LLM.
from dotenv import dotenv_values

dotenv_values()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
#Step 1 - setup OCI Generative AI llm



# use default authN method API-key
llm = GoogleGenerativeAI(
    model = "gemini-1.5-flash",google_api_key=os.environ["GOOGLE_API_KEY"]
)

#Step 2 - here we connect to a chromadb server. we need to run the chromadb server before we connect to it

client = chromadb.HttpClient(host="127.0.0.1", port=8000)  # Specify port if needed


#Step 3 - here we crete embeddings using 'cohere.embed-english-light-v2.0" model.

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

#Step 4 - here we create a retriever that gets relevant documents (similar in meaning to a query)

db = Chroma(client=client, embedding_function=embeddings)

retv = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

#Step 5 - here we can explore how similar documents to the query are returned by prining the document metadata. This step is optional

docs = retv.invoke('Tell us what is RAG Fusion')

def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )

pretty_print_docs(docs)

for doc in docs:
    print(doc.metadata)

#Step 6 - here we create a retrieval chain that takes llm , retirever objects and invoke it to get a response to our query

chain = RetrievalQA.from_chain_type(llm=llm, retriever=retv,return_source_documents=True)

response = chain.invoke("Tell us which module is most relevant to LLMs and Generative AI")

print(response)


