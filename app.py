import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from tutor.prompt import SYSTEM_PROMPT
from tutor.rag_chain import generate_answer

st.set_page_config(
    page_title="Class 4 Math Tutor",
    layout="wide"
)

st.title("📚 Personalized AI Math Tutor")

student_name = st.text_input(
    "Student Name"
)

student_level = st.selectbox(
    "Level",
    [
        "Beginner",
        "Intermediate",
        "Advanced"
    ]
)

question = st.text_input(
    "Ask your Math Question"
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    "vector_db",
    embeddings,
    allow_dangerous_deserialization=True
)

if st.button("Get Answer"):

    docs = db.similarity_search(
        question,
        k=5
    )

    context = "\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = SYSTEM_PROMPT.format(
        context=context,
        name=student_name,
        level=student_level,
        question=question
    )

    answer = generate_answer(prompt)

    st.subheader("Answer")

    st.write(answer)
