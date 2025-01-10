import chromadb
from langchain.load import dumps , loads
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatMessagePromptTemplate,ChatPromptTemplate
from langchain.prompts import SystemMessagePromptTemplate,HumanMessagePromptTemplate,ChatMessagePromptTemplate,PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.chains import RetrievalQA
from chromadb.config import Settings
from dotenv import load_dotenv
from pprint import pprint
import os


os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

llm = GoogleGenerativeAI(
    model = "gemini-1.5-flash",google_api_key=os.environ["GOOGLE_API_KEY"])


client = chromadb.HttpClient(host="127.0.0.1", settings=Settings(allow_reset=True))
db = Chroma(client= client, embedding_function=embeddings)


retv = db.as_retriever()


prompt =   ChatPromptTemplate(
                            input_variables=['original_query'], 
                            messages=[
                                    SystemMessagePromptTemplate(
                                                            prompt=PromptTemplate(
                                                                                input_variables=[], 
                                                                                template='You are a helpful assistant that generates multiple search queries based on a single input query.'
                                                                                )
                                                                ), 
                                    HumanMessagePromptTemplate(
                                                            prompt=PromptTemplate(
                                                                                input_variables=['original_query'], 
                                                                                template='Generate multiple search queries related to: {question} \n OUTPUT (4 queries):'
                                                                                )
                                                                )
                                    ])





generate_queries = (
    prompt|
    llm|
    StrOutputParser()|
    (lambda x: x.split("\n"))

)


def reciprocal_rank_fusion(results:list[list],k=10):
    fused_scores = {}
    for docs in results:
        for rank,doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str]=0
            fused_scores[doc_str] += 1/ (rank+k)

    reranked_results = [
        (loads(doc),score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    return reranked_results


ragfusion_chain = generate_queries | retv.map()|reciprocal_rank_fusion


query = ragfusion_chain.invoke({"question": "What is an rag fusion system?"})

schema =ragfusion_chain.input_schema.model_json_schema()
""" pprint(schema)
print("\n\n\n\n\n")
pprint(query)
 """


template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

full_rag_fusion_chain = (
                        {
                            "context": ragfusion_chain,
                            "question": RunnablePassthrough()
                        }
                        | prompt
                        | llm
                        | StrOutputParser()
                        )


result =full_rag_fusion_chain.invoke({"question": "Tell me about retreival augmented generation??"})


print(result)