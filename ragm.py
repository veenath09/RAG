from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts  import ChatPromptTemplate
from langchain.memory.buffer import ConversationBufferMemory
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationalRetrievalChain
import os
from pprint import  pprint
import chromadb
from dotenv import dotenv_values

dotenv_values()

#setup connection with chromadb
client = chromadb.HttpClient(host="127.0.0.1", port=8000)


#setup API key
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

#embeddings model
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

#setting up the LLM

llm = GoogleGenerativeAI(model="gemini-1.5-flash",
    google_api_key=os.environ["GOOGLE_API_KEY"]
)

#setup the databse

vectorstore = Chroma(client=client , embedding_function=embeddings)


#setting up the retriver

retriever = vectorstore.as_retriever(search_type ="similarity", search_kwargs={"k":3})

memory = ConversationBufferMemory( memory_key="chat_history", return_messages=True,output_key='answer')


qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever= retriever,
    memory= memory,
    return_source_documents = True
)


response = qa.invoke({"question": "Tell me about RAG"})
print(memory.chat_memory.messages)
print("\n\n\n\n")

response = qa.invoke({"question": "Which module of the paper is relavent to the Logits-based Fusion"})
print(memory.chat_memory.messages)
print("\n\n\n\n")
print(response)