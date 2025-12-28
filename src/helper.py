from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
def load_pdf_documents(file_path):
    loader = DirectoryLoader(
    file_path,
    glob="**/*.pdf",
    loader_cls=PyPDFLoader,

)
    
    documents=loader.load()
    return documents

document=load_pdf_documents("E:\\AI-Buildathon\\data")

def text_splitter(documents):
    # Optimized for medical/clinical content with semantic boundaries
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # Smaller chunks for better precision
        chunk_overlap=200,  # More overlap to preserve context
        length_function=len,
        separators=["\n\n", "\n", ". ", ", ", " ", ""],  # Preserve sentence boundaries
        is_separator_regex=False,
    )
    text_chunk = text_splitter.split_documents(documents)
    
    # Add chunk metadata for better tracking
    for i, chunk in enumerate(text_chunk):
        chunk.metadata['chunk_id'] = i
        chunk.metadata['chunk_size'] = len(chunk.page_content)
    
    return text_chunk

text_chunk = text_splitter(document)
print(f"Number of Chunks: {len(text_chunk)}")
print(f"Average Chunk Size: {sum(c.metadata['chunk_size'] for c in text_chunk) / len(text_chunk):.0f} characters") 


# Using a more powerful embedding model optimized for semantic understanding
# Options: 
# 1. "sentence-transformers/all-mpnet-base-v2" - Better semantic understanding (768 dim)
# 2. "BAAI/bge-base-en-v1.5" - State-of-the-art retrieval performance (768 dim)
# 3. "sentence-transformers/multi-qa-mpnet-base-dot-v1" - Optimized for Q&A (768 dim)

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={'device': 'cpu'},  # Change to 'cuda' if GPU available
    encode_kwargs={'normalize_embeddings': True}  # Improves retrieval quality
)
