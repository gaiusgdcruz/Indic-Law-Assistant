# Indic-Law-Assistant

A Retrieval-Augmented Generation (RAG) system designed for Indian legal documents using local Large Language Models via Ollama. It parses Law Document PDFs, indexes them using ChromaDB, and enables semantic query answering via a Streamlit frontend and FastAPI.

---

## ✨ Prerequisites

Ensure the following are installed before running the project:

- Python 3.10+
- `pip` for dependency management
- [Ollama](https://ollama.com/) (running locally for LLM inference)
- Chrome or any browser for Streamlit

### 1. Create a Virtual Environment
Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 2. Install Ollama
Begin by installing [Ollama](https://ollama.com/) following the instructions provided on the website.

### 3. Pull the Model
Pull the required model using the command, for example for `deepseek-r1:1.5b`:
```bash
ollama pull deepseek-r1:1.5b
```

### 4. Start Ollama
Once Ollama is installed, start the Ollama service:
```bash
ollama serve
```


## 🚀 FastAPI Setup

1. **Clone the repository**

2. **Activate your virtual environment** (if not already):
```bash
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

3. **Navigate to the API folder**:
```bash
cd api/
```

4. **Install dependencies**:
```bash
pip install -r requirements.txt
```

5. **Run the FastAPI server**:
```bash
uvicorn main:app --reload
```

Visit the API docs at:
```
http://localhost:8000/docs
```

### 🔍 Query Endpoint

- **Endpoint**: `/rag_query`
- **Method**: POST
- **Payload**:
```json
{
  "query": "What is Indian Forest law?"
}
```
- **Response**:
```json
{
  "answer": "The Indian forest law is...",
  "retrieved_sources": [...],
  "rewritten_query": "...",
  "processing_time": 2.34,
  "confidence_score": 0.91
}
```

## 💬 Streamlit UI Setup

1. **Navigate to the Streamlit UI folder**:
```bash
cd streamlit_ui/
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the Streamlit app**:
```bash
streamlit run main.py
```

The app should launch in your browser for interactive legal queries.
