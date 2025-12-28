from src.helper import load_pdf_documents, text_splitter, HuggingFaceEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os
load_dotenv("E:\\AI-Buildathon\\.env")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index("test")


extracted_data=load_pdf_documents("E:\\AI-Buildathon\\data")
text_chunks=text_splitter(extracted_data)
embeddings= HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={'device': 'cpu'},  # Change to 'cuda' if GPU available
    encode_kwargs={'normalize_embeddings': True}  # Improves retrieval quality
)



docstore = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name="test",
)
docsearch= PineconeVectorStore.from_existing_index(
    embedding=embeddings,
    index_name="test"
)

