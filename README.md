# 📄 LLM Question Answering App (RAG Project)

This project is a Retrieval-Augmented Generation (RAG) application that allows users to upload documents and ask questions based on their content using Large Language Models (LLMs).

---

## 🚀 Features

- Upload documents (PDF, DOCX, TXT)
- Automatic text chunking
- Generate embeddings using OpenAI
- Store embeddings in ChromaDB
- Ask questions about uploaded documents
- Get accurate, context-based answers
- Chat history tracking

---

## 🛠️ Tech Stack

- Frontend: Streamlit
- LLM: OpenAI
- Framework: LangChain
- Vector DB: ChromaDB
- Language: Python

---

## 📁 Project Structure

RAG PROJECT/
│── chat_with_document.py
│── requirements.txt
│── img.webp
│── Formula1.pdf (sample file)
│── .env (not included in repo)
│── .gitignore

---

## ⚙️ Installation

1. Clone the repository
git clone https://github.com/your-username/rag-project.git
cd rag-project

2. Create virtual environment
python -m venv venv
venv\Scripts\activate

3. Install dependencies
pip install -r requirements.txt

4. Add OpenAI API Key
Create a .env file and add:
OPENAI_API_KEY=your_api_key_here

5. Run the app
streamlit run chat_with_document.py

---

## 🧠 How It Works

1. User uploads a document
2. Text is split into chunks
3. Embeddings are created using OpenAI
4. Stored in Chroma vector database
5. User asks a question
6. Relevant chunks are retrieved
7. LLM generates an answer based on context

---

## 📸 Demo

(Add screenshots here later)

---

## 🎯 Use Cases

- Document-based Q&A systems
- Research assistants
- Chat with PDFs
- Knowledge base querying

---

## 🔒 Security Note

- .env file is excluded to protect API keys
- Never expose your API keys publicly

---

## 📌 Future Improvements

- Chat UI (like ChatGPT)
- Multi-file support
- Deployment (Streamlit Cloud / Render)
- Authentication system

---

## 👨‍💻 Author

Harshit Kumar Mehta

---

## ⭐ If you like this project

Give it a star on GitHub!