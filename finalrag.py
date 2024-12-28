from fastapi import FastAPI, Form
import chromadb
from langchain_community.vectorstores import Chroma
from chromadb.config import Settings
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()


from uuid import uuid4


unique_id = uuid4().hex[0:8]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"Test - {unique_id}"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com/"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")

# Step 1: Authenticate using the "GOOGLE_API_KEY"
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Step 2: Create a custom chain to balance LLM knowledge and retrieved context
def create_chain():
    # Initialize Chroma client and embedding function
    client = chromadb.HttpClient(host="127.0.0.1", settings=Settings(allow_reset=True))
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = Chroma(client=client, embedding_function=embeddings)


    # Retrieve documents with MMR (Maximal Marginal Relevance)
    retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 5})
    
    # Initialize the LLM
    llm = GoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.environ["GOOGLE_API_KEY"]
    )
    
    # Define a custom prompt
    custom_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        You are an assistant. Answer the user's question by combining knowledge from the context provided and your own expertise.
        Be informative, complete, and go beyond the context if needed.

        Context: {context}
        User Question: {question}
        
        Answer:
        """
    )
    
    # Create the RetrievalQA chain with the custom prompt
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": custom_prompt},
        return_source_documents=True
    )
    
    return qa_chain

# Step 3: Instantiate the chain
chain = create_chain()

# Step 4: Define the chat function
def chat(user_message):
    # Generate a response using the chain
    bot_response = chain({"query": user_message})
    
    # Structure the bot's response and return
    response = {
        "question": user_message,
        "answer": bot_response.get("result", ""),
        "sources": [
            doc.metadata.get("source", "Unknown source") for doc in bot_response.get("source_documents", [])
        ],
    }
    return response

# Step 5: Setup Streamlit UI
if __name__ == "__main__":
    import streamlit as st

    st.title("RAG GENAI Chat BOT ")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    user_input = st.chat_input("Type your question here...")
    if user_input:
        bot_response = chat(user_input)
        
        # Save conversation history
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append({"role": "assistant", "content": bot_response["answer"]})
        
        # Display the conversation
        for i, message in enumerate(st.session_state.messages):
            if message["role"] == "user":
                st.chat_message("user").write(f"Question: {message['content']}")
            elif message["role"] == "assistant":
                st.chat_message("assistant").write(f"Answer: {message['content']}")
                st.write("Sources:", bot_response.get("sources", ["No sources found."]))
