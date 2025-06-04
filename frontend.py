import os
import streamlit as st
from rag_pipeline import answer_query, retrieve_docs, llm_model
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq

st.set_page_config(page_title="AI Lawyer", layout="wide")
st.title("⚖️ AI-Powered Legal Assistant")

# --- PDF Upload ---
uploaded_file = st.file_uploader("📄 Upload a legal PDF", type=["pdf"], accept_multiple_files=False)

# --- Summarization ---
def summarize_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    full_text = " ".join([p.page_content for p in pages])
    full_text = full_text[:4000]  # truncate if needed

    prompt_template = PromptTemplate.from_template(
        "Summarize this legal document in plain English:\n\n{document}"
    )

    llm = ChatGroq(
        model="deepseek-r1-distill-llama-70b",
        api_key=os.getenv("GROQ_API_KEY")
    )

    chain = LLMChain(llm=llm, prompt=prompt_template)
    summary = chain.invoke({"document": full_text})
    return summary["text"]

# --- Step 1: If file uploaded, show summary ---
if uploaded_file:
    with st.spinner("📚 Summarizing the document..."):
        temp_path = f"./temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        summary_text = summarize_pdf(temp_path)
        st.subheader("📌 Document Summary")
        st.success(summary_text)
else:
    st.info("Upload a PDF to get started.")

# --- Step 2: Chat UI ---
st.divider()
user_query = st.text_area("🧠 Enter your legal question", height=150, placeholder="Ask anything about the uploaded document...")
ask_question = st.button("Ask AI Lawyer")

if ask_question:
    if uploaded_file:
        st.chat_message("user").write(user_query)

        # RAG Pipeline
        retrieved_docs = retrieve_docs(user_query)
        response = answer_query(documents=retrieved_docs, model=llm_model, query=user_query)

        st.chat_message("AI Lawyer").write(response)

    else:
        st.error("⚠️ Please upload a PDF before asking questions.")
