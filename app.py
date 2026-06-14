import streamlit as st
import os

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_huggingface import HuggingFaceEndpoint

# -----------------------------------
# PAGE CONFIG
# -----------------------------------

st.set_page_config(
    page_title="Class 4 AI Math Tutor",
    layout="wide"
)

st.title("📚 Personalized AI Math Tutor")

# -----------------------------------
# STUDENT PROFILE
# -----------------------------------

student_name = st.text_input(
    "Student Name",
    "Student"
)

student_level = st.selectbox(
    "Level",
    [
        "Beginner",
        "Intermediate",
        "Advanced"
    ]
)

# -----------------------------------
# EMBEDDINGS
# -----------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    "vector_db",
    embeddings,
    allow_dangerous_deserialization=True
)

# -----------------------------------
# MEMORY
# -----------------------------------

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# -----------------------------------
# LLM
# -----------------------------------

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    huggingfacehub_api_token=os.getenv(
        "HUGGINGFACEHUB_API_TOKEN"
    ),
    temperature=0.3,
    max_new_tokens=512
)

# -----------------------------------
# CHAIN
# -----------------------------------

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=db.as_retriever(
        search_kwargs={"k": 5}
    ),
    memory=memory
)

# -----------------------------------
# CHAT
# -----------------------------------

question = st.chat_input(
    "Ask a Math Question"
)

if question:

    prompt = f"""
You are an expert CBSE Class 4 Mathematics Teacher.

Student Name:
{student_name}

Level:
{student_level}

Answer in child-friendly language.

Question:
{question}
"""

    response = qa_chain.invoke(
        {
            "question": prompt
        }
    )

    st.chat_message("user").write(
        question
    )

    st.chat_message("assistant").write(
        response["answer"]
    )
