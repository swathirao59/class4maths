import os

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


PDF_FOLDER = "pdfs"

documents = []

for file in os.listdir(PDF_FOLDER):

    if file.endswith(".pdf"):

        loader = PyPDFLoader(
            os.path.join(PDF_FOLDER, file)
        )

        documents.extend(loader.load())

print(f"Loaded {len(documents)} pages")


splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)

print("Chunks:", len(chunks))

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = FAISS.from_documents(
    chunks,
    embeddings
)

vector_db.save_local("vector_db")

print("Vector DB Created Successfully")
