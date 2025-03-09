import os
from src.prompt import *
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from src.helper import download_hugging_face_embeddings
from flask import Flask,render_template,jsonify,request
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

app = Flask(__name__)

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
INDEX_NAME = "testindex"

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/chat",methods=["GET","POST"])
def chat():
    msg = request.form["msg"]
    rag_chain = rag()
    response = rag_chain.invoke(
        {
            "input":msg
        }
    )
    return str(
        response["answer"]
    )

def download_embeddings():
    embeddings = download_hugging_face_embeddings()
    return embeddings

def initialize_vector_Store(embeddings):
    docsearch = PineconeVectorStore.from_existing_index(
            index_name = INDEX_NAME,
            embedding=embeddings
        )
    return docsearch

def retriever_initializer(docsearch):
    retriever = docsearch.as_retriever(search_type="similarity",search_kwargs={"k":5})
    return retriever

def llm_Setup():
    llm = OpenAI(
        temperature=0.4,
        max_tokens=500
        )
    return llm

def prepare_prompt(prompt_temp):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",prompt_temp),
            ("human","{input}")
        ]
    )
    return prompt

def create_qa_rag_chain(retriever,llm,prompt):
    question_answer_chain = create_stuff_documents_chain(llm,prompt)
    rag_chain = create_retrieval_chain(retriever,question_answer_chain)
    return rag_chain

def rag():
    embeddings = download_embeddings()
    doc_search = initialize_vector_Store(embeddings=embeddings)
    retriever = retriever_initializer(docsearch=doc_search)
    llm = llm_Setup()
    prompt = prepare_prompt(prompt_temp=prompt_template)
    rag_chain = create_qa_rag_chain(retriever,llm,prompt)
    return rag_chain

if __name__ == '__main__':
    app.run()