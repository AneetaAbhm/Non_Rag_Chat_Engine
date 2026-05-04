import logging
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from answer_questions import handle_respons
from processing_file import (
    get_text_pdf,
    get_text_image,
    build_context,
    MAX_DOCS,
    MAX_PAGES_PER_DOC,
)

logging.basicConfig(
    format="%(asctime)s %(message)s",
    level=logging.INFO,
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()]
)


def main():
    load_dotenv()

    st.title("📄 Ask Your Documents (PDF & Images)")
    st.write(
        f"Upload up to **{MAX_DOCS} files** (PDF or image), max **{MAX_PAGES_PER_DOC} pages** each. "
    )

    # ── Sidebar — Model Switcher ───────────────────────────────────────────────
    with st.sidebar:
        st.title("⚙️ Settings")

        model_options = {
            "Llama 3.1 8B (Fast)":        "llama-3.1-8b-instant",
            "Llama 3.3 70B (Smart)":       "llama-3.3-70b-versatile",
            "Llama 4 Scout 17B (Preview)": "meta-llama/llama-4-scout-17b-16e-instruct",
            "Qwen3 32B (Preview)":         "qwen/qwen3-32b",
        }

        selected_label = st.selectbox(
            "Select Model",
            list(model_options.keys()),
            index=0
        )
        selected_model = model_options[selected_label]
        st.info(f"Using: `{selected_model}`")

    # Session state defaults
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("context", None)
    st.session_state.setdefault("docs_processed", False)
    st.session_state.setdefault("llm", None)
    st.session_state.setdefault("current_model", None)

    # ── File upload widgets ────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        pdf_files = st.file_uploader(
            "Upload PDF(s)",
            type=["pdf"],
            accept_multiple_files=True,
            key="pdf_uploader"
        )
    with col2:
        image_files = st.file_uploader(
            "Upload Image(s)",
            type=["png", "jpg", "jpeg", "tiff", "bmp"],
            accept_multiple_files=True,
            key="img_uploader"
        )

    all_files = (pdf_files or []) + (image_files or [])

    if len(all_files) > MAX_DOCS:
        st.error(f"❌ Too many files. Please upload a maximum of {MAX_DOCS} documents in total.")
        return

    # ── Process button ─────────────────────────────────────────────────────────
    if all_files and st.button("📥 Process Documents"):
        with st.spinner("Extracting text from documents..."):
            documents = []

            if pdf_files:
                documents += get_text_pdf(pdf_files)
                # Warn if any PDF exceeded page limit
                from PyPDF2 import PdfReader
                for pdf in pdf_files:
                    reader = PdfReader(pdf)
                    if len(reader.pages) > MAX_PAGES_PER_DOC:
                        st.warning(f"⚠️ '{pdf.name}' has more than {MAX_PAGES_PER_DOC} pages. Only first {MAX_PAGES_PER_DOC} pages were used.")

            if image_files:
                documents += get_text_image(image_files)

            if not documents:
                st.error("Could not extract text from any uploaded file.")
                return

            context = build_context(documents)
            st.session_state.context = context
            st.session_state.docs_processed = True
            st.session_state.chat_history = []
            st.session_state.current_model = selected_model

            # ── Groq LLM ──────────────────────────────────────────────────
            groq_api_key = os.getenv("GROQ_API_KEY")
            if not groq_api_key:
                st.error("❌ GROQ_API_KEY not found. Please add it to your .env file.")
                return

            st.session_state.llm = ChatGroq(
                model=selected_model,
                temperature=0.1,
                api_key=groq_api_key
            )

            st.success(f"✅ Processed {len(documents)} document(s) with **{selected_label}**. You can now ask questions!")

            with st.expander("🔍 View extracted context (debug)"):
                st.text(context[:3000] + ("..." if len(context) > 3000 else ""))

    # If model switched after processing, update LLM silently
    if (st.session_state.docs_processed and
            st.session_state.current_model != selected_model):
        groq_api_key = os.getenv("GROQ_API_KEY")
        st.session_state.llm = ChatGroq(
            model=selected_model,
            temperature=0.1,
            api_key=groq_api_key
        )
        st.session_state.current_model = selected_model
        st.toast(f"🔄 Switched to {selected_label}", icon="✅")

    if not all_files and not st.session_state.docs_processed:
        st.info("Please upload at least one PDF or image file, then click **Process Documents**.")

    # ── Chat input ─────────────────────────────────────────────────────────────
    if st.session_state.docs_processed:
        user_question = st.chat_input("Ask a question about your documents...")
        if user_question:
            handle_respons(user_question)


if __name__ == "__main__":
    main()