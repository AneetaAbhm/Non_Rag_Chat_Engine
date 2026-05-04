import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


SYSTEM_PROMPT = """You are a helpful document assistant. You will be given the full text of one or more uploaded documents, followed by a user question.

Rules:
- Answer ONLY based on the provided document content. Do NOT use any outside knowledge.
- If the answer is not found in the documents, say: "I could not find that information in the provided documents."
- Be concise and accurate.
- When quoting from the document, mention which document it came from.
- You have access to prior conversation history to handle follow-up questions.
"""

# Keep only last N messages to avoid token overflow
HISTORY_LIMIT = 6  # = 3 Q&A pairs


def build_messages(context: str, chat_history: list, user_question: str) -> list:
    """
    Build the message list to send to the LLM:
      [SystemMessage] → [last HISTORY_LIMIT messages] → [HumanMessage with context+question]
    """
    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    for msg in chat_history:
        messages.append(msg)

    user_content = f"""=== DOCUMENT CONTEXT ===
{context}
=== END OF DOCUMENTS ===

Question: {user_question}"""

    messages.append(HumanMessage(content=user_content))
    return messages


def handle_respons(user_question: str):
    """Main handler called from main.py when the user submits a question."""

    if not st.session_state.get("context"):
        st.error("No document context found. Please upload and process documents first.")
        return

    if 'chat_history' not in st.session_state or st.session_state.chat_history is None:
        st.session_state.chat_history = []

    # Render existing chat history
    for message in st.session_state.chat_history:
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        with st.chat_message(role):
            display_text = message.content
            if isinstance(message, HumanMessage) and "=== DOCUMENT CONTEXT ===" in display_text:
                display_text = display_text.split("Question:")[-1].strip()
            st.write(display_text)

    # Show current user message
    with st.chat_message("user"):
        st.write(user_question)

    # Limit history to last HISTORY_LIMIT messages before passing to LLM
    recent_history = st.session_state.chat_history[-HISTORY_LIMIT:]

    clean_history = []
    for msg in recent_history:
        if isinstance(msg, HumanMessage):
            q = msg.content
            if "Question:" in q:
                q = q.split("Question:")[-1].strip()
            clean_history.append(HumanMessage(content=q))
        else:
            clean_history.append(msg)

    messages = build_messages(
        context=st.session_state.context,
        chat_history=clean_history,
        user_question=user_question
    )

    # Call the LLM
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                llm: ChatGroq = st.session_state.llm
                response = llm.invoke(messages)
                assistant_reply = response.content
            except Exception as e:
                st.error(f"LLM error: {e}")
                assistant_reply = "Sorry, I encountered an error. Please try again."

        st.write(assistant_reply)

    # Save to history
    st.session_state.chat_history.append(
        HumanMessage(content=f"=== DOCUMENT CONTEXT ===\n{st.session_state.context}\n=== END OF DOCUMENTS ===\n\nQuestion: {user_question}")
    )
    st.session_state.chat_history.append(AIMessage(content=assistant_reply))