from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

import streamlit as st
import os

# Load from .env locally, Streamlit secrets in production
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)

def load_document(file):
    name, extension = os.path.splitext(file)

    if extension == '.pdf':
        from langchain_community.document_loaders import PyMuPDFLoader
        print(f'Loading {file}')
        loader = PyMuPDFLoader(file)

    elif extension == '.docx':
        from langchain_community.document_loaders import Docx2txtLoader
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)

    else:
        print('Document format is not supported!')
        return None

    data = loader.load()
    return data


def chunk_data(data, chunk_size=256, chunk_overlap=100):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks


def create_embeddings(chunks):
    import uuid
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(
        chunks, 
        embeddings,
        collection_name=f"rag_{uuid.uuid4().hex}"
    )
    return vector_store


def ask_and_get_answer(vector_store, query, k=3):
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    retriever = vector_store.as_retriever(search_kwargs={"k": k})

    prompt = ChatPromptTemplate.from_template("""
    Answer the question based only on the context below:

    {context}

    Question: {question}
    """)

    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    chain = prompt | llm
    for chunk in chain.stream({"context": context, "question": query}):
        yield chunk.content


def calculate_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-3-small')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    return total_tokens, total_tokens / 1000 * 0.0004


def clear_history():
    if 'messages' in st.session_state:
        del st.session_state['messages']


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY not found in environment or .env file. Please add it to your .env file.")
        st.stop()

    st.image('img.webp', width=400)
    st.subheader('LLM Question-Answering Application 🤖')

    with st.sidebar:
        uploaded_file = st.file_uploader('Upload a file:', type=['pdf', 'docx', 'txt'])
        chunk_size = st.number_input('Chunk size:', min_value=100, max_value=2048, value=1024, on_change=clear_history)
        k = st.number_input('k(number of chunks)', min_value=1, max_value=20, value=3, on_change=clear_history)

        add_data = st.button('Add Data', on_click=clear_history)

        if uploaded_file and add_data:
            with st.spinner('Reading, chunking and embedding file ...'):
                bytes_data = uploaded_file.read()
                file_name = os.path.join('./', uploaded_file.name)

                with open(file_name, 'wb') as f:
                    f.write(bytes_data)

                data = load_document(file_name)
                chunks = chunk_data(data, chunk_size=chunk_size)
                st.write(f'Chunk size: {chunk_size}, Chunks: {len(chunks)}')

                tokens, embedding_cost = calculate_embedding_cost(chunks)
                st.write(f'Embedding cost: ${embedding_cost:.4f}')

                vector_store = create_embeddings(chunks)
                st.session_state.vs = vector_store
                st.success('Data added successfully!')

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Clean up any empty messages from interrupted runs
    st.session_state.messages = [msg for msg in st.session_state.messages if msg["content"]]

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if query := st.chat_input('Ask a question about the document:'):
        if 'vs' in st.session_state:
            vector_store = st.session_state.vs
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(query)
            st.session_state.messages.append({"role": "user", "content": query})

            # Stream assistant response inside the native chat bubble
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.markdown("▌ *Thinking...*")
                
                stop_placeholder = st.empty()
                stop_placeholder.button("🛑 Stop generating")
                
                full_response = ""
                st.session_state.messages.append({"role": "assistant", "content": ""})
                
                stream = ask_and_get_answer(vector_store, query, k)
                    
                for chunk in stream:
                    full_response += chunk
                    st.session_state.messages[-1]["content"] = full_response
                    message_placeholder.markdown(full_response + "▌")
                    
                message_placeholder.markdown(full_response)
                stop_placeholder.empty()
            
        else:
            st.warning("Please upload a document and add data first.")