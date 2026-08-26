import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def load_documents():
    """data folder se saare PDFs load karta hai"""
    loader = DirectoryLoader('data/', glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f"Total documents loaded: {len(documents)}")
    return documents

def split_documents(documents):
    """Documents ko chote chunks me todta hai"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")
    return chunks

def create_vector_store(chunks):
    """Chunks ko embeddings me convert karke ChromaDB me store karta hai"""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )
    print("Vector store created and saved successfully!")
    return vectorstore

if __name__ == "__main__":
    docs = load_documents()
    chunks = split_documents(docs)
    create_vector_store(chunks)