import os
import logging
from src.prompt import *
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from src.helper import download_hugging_face_embeddings
from flask import Flask, render_template, jsonify, request
from langchain.chains.combine_documents import create_stuff_documents_chain
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
PINECONE_API_KEY = os.environ['PINECONE_API_KEY']
INDEX_NAME = "testindex"

@app.route('/')
def index():
    logger.info("Rendering index page")
    return render_template('chat.html')

@app.route("/chat", methods=["GET", "POST"])
def chat():
    try:
        msg = request.form["msg"]
        logger.info(f"Received chat message: {msg}")
        response = rag_chain.invoke({"input": msg})
        logger.info(f"Generated response: {response['answer']}")
        return str(response["answer"])
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@lru_cache(maxsize=1)
def download_embeddings():
    logger.info("Downloading Hugging Face embeddings")
    embeddings = download_hugging_face_embeddings()
    return embeddings

def initialize_vector_Store(embeddings):
    logger.info(f"Initializing Pinecone vector store with index: {INDEX_NAME}")
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME,
        embedding=embeddings
    )
    return docsearch

def retriever_initializer(docsearch):
    logger.info("Initializing retriever with similarity search")
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    return retriever

def llm_Setup():
    logger.info("Setting up OpenAI LLM")
    llm = OpenAI(temperature=0.4, max_tokens=500)
    return llm

def prepare_prompt(prompt_temp):
    logger.info("Preparing chat prompt template")
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_temp),
        ("human", "{input}")
    ])
    return prompt

def create_qa_rag_chain(retriever, llm, prompt):
    logger.info("Creating QA RAG chain")
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    return rag_chain


logger.info("Initializing RAG pipeline")
embeddings = download_embeddings()

doc_search = initialize_vector_Store(embeddings=embeddings)
retriever = retriever_initializer(docsearch=doc_search)
llm = llm_Setup()
prompt = prepare_prompt(prompt_temp=prompt_template)
rag_chain = create_qa_rag_chain(retriever, llm, prompt)


if __name__ == '__main__':
    logger.info("Starting Flask application")
    app.run(host="0.0.0.0", port=8080, debug=True)