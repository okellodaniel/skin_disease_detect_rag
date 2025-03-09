from langchain_pinecone import PineconeVectorStore
from src.helper import load_pdf,download_hugging_face_embeddings,text_split
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY

INDEX_NAME = "testindex"

pc = Pinecone(api_key=PINECONE_API_KEY)


extracted_data = load_pdf(data='data/')
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()


# create index in pinecone cloud
if not pc.has_index(INDEX_NAME):
    pc.create_index(
        name=   INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

# Embed each chunk and upsert the embeddings into pinecone index
docsearch = docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name= INDEX_NAME,
    embedding=embeddings
)