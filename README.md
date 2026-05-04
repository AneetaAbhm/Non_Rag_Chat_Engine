# 📄 Chat Engine — Non-RAG Document Q&A

A conversational chat engine that reads PDF and image documents directly into an LLM's context window and answers user queries — **without any vector database or RAG pipeline**.

---

## 🚀 Features

- 📄 Upload up to **3 documents** (PDF or image)
- 🖼️ OCR support for image files (PNG, JPG, TIFF, BMP)
- 🧠 Direct in-context LLM reading — no embeddings, no vector DB
- 💬 Conversation memory — last 3 Q&A pairs maintained
- ✂️ Smart token handling — stopword removal + word count truncation
- 🔄 Model switcher — choose from multiple Groq-hosted LLMs
- ⚡ Fast responses via Groq cloud API

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| UI | Streamlit |
| PDF Parsing | PyPDF2 |
| Image OCR | Tesseract + pytesseract |
| Stopword Removal | NLTK |
| LLM | Groq API (Llama, Qwen) |
| LangChain | langchain-groq, langchain-core |
| Env Management | python-dotenv |

---

## 📁 Project Structure

```
chat-engine/
│
├── main.py               # Streamlit UI + file upload + LLM init
├── processing_file.py    # PDF parsing, OCR, stopword removal, context building
├── answer_questions.py   # Prompt engineering + LLM call + chat history
├── requirements.txt      # Python dependencies
└── .env                  # API keys (not committed to git)
```

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/chat-engine.git
cd chat-engine
```

### 2. Install Tesseract OCR (system binary)
```bash
# Windows — download from:
# https://github.com/UB-Mannheim/tesseract/wiki

# Ubuntu/Debian
sudo apt install tesseract-ocr

# macOS
brew install tesseract
```

### 3. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```

Get your free Groq API key at: https://console.groq.com

---

## ▶️ Running the App

```bash
streamlit run main.py
```

Open your browser at `http://localhost:8501`

---

## 📌 Input Constraints

| Constraint | Limit |
|---|---|
| Max documents per session | 3 |
| Max pages per PDF | 5 |
| Supported PDF format | Text-based PDF |
| Supported image formats | PNG, JPG, JPEG, TIFF, BMP |
| Token budget | ~3000 tokens |

---

## 🧠 How It Works

```
User uploads PDF / Image
        ↓
Text Extraction
(PyPDF2 for PDF, Tesseract OCR for images)
        ↓
Stopword Removal
(Remove safe filler words, preserve negations)
        ↓
Context Building + Token Check
(Combine text, truncate if over word budget)
        ↓
Prompt Engineering
(System prompt + chat history + context + question)
        ↓
Groq LLM API
        ↓
Answer displayed in chat UI
```

---

## 🤖 Available Models

| Model | Best For |
|---|---|
| Llama 3.1 8B Instant | Fast, simple Q&A |
| Llama 3.3 70B Versatile | Complex reasoning |
| Llama 4 Scout 17B | Latest, preview |
| Qwen3 32B | Strong reasoning, preview |

---

## 📦 Requirements

```
streamlit
PyPDF2
Pillow
pytesseract
langchain-groq
langchain-core
nltk
python-dotenv
```

---

## ⚠️ Important Notes

- **No RAG used** — document content is passed directly into the LLM context
- **No vector database** — no FAISS, no embeddings anywhere in the pipeline
- The Tesseract path in `processing_file.py` is set for Windows. Update if using Linux/macOS:
  ```python
  # Linux/macOS — comment out or update this line:
  pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
  ```

---

## 📄 License

MIT License
