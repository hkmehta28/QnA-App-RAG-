import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import os

def load_document(file):
    import os
    name, extension = os.path.splitext(file)
    
    if extension == '.pdf':
        from langchain_community.document_loaders import PyPDFLoader
        print(f'Loading {file}')
        loader = PyPDFLoader(file)

    elif extension == '.docx':
        from langchain_community.document_loaders import Docx2txtLoader
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)

    else:
        print('Document format is not supported!')
        return None
        
    data = loader.load()
    return data


def chunk_data(data, chunk_size = 256, chunk_overlap=20):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    chunks = text_splitter.split_documents(data)
    return chunks


def create_embeddings(chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(chunks, embeddings)
    return vector_store


def ask_and_get_answer(vector_store, query, k=3):
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate

    # LLM
    llm = ChatOpenAI(model="gpt-4.1-mini")

    # Retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # Prompt
    prompt = ChatPromptTemplate.from_template("""
    Answer the question based only on the context below:

    {context}

    Question: {question}
    """)

    # Get relevant docs
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Final response
    response = llm.invoke(prompt.format(context=context, question=query))

    return response.content


def calculate_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    # print(f'Total Tokens: {total_tokens}')
    # print(f'Embedding Cost in USD: {(total_tokens / 1000) * 0.0004:.6f}')
    return total_tokens, total_tokens / 1000 * 0.0004

def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']


if __name__ == "__main__":
    import os

    st.image('img.webp', width=400)
    st.subheader('LLM Question-Answering Application 🤖')

    with st.sidebar:
        api_key = st.text_input('OpenAI API Key:', type='password')
        if not api_key:
            st.warning("Please enter your OpenAI API Key to continue")
            st.stop()
        os.environ['OPENAI_API_KEY'] = api_key

        uploaded_file = st.file_uploader('Upload a file:', type=['pdf', 'docx', 'txt'])
        chunk_size = st.number_input('Chunk size:', min_value=100, max_value=2048, value=512, on_change=clear_history)
        k = st.number_input('k', min_value=1, max_value=20, value=3, on_change=clear_history)

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

    query = st.text_input('Ask a question about the document:')
    if query:
       
       if 'vs' in st.session_state:
            vector_store = st.session_state.vs
            st.write(f'k: {k}')
            
            answer = ask_and_get_answer(vector_store, query, k)
            st.text_area('LLM Answer:', value=answer)

    # else:
    # st.warning('Please upload a document and add data first!')
            st.divider()

            if 'history' not in st.session_state:
                st.session_state.history = ''

            value = f'Q: {query} \nA: {answer}'

            st.session_state.history = (
                f'{value} \n {"-" * 100} \n {st.session_state.history}'
            )

            st.text_area(label='Chat History', key='history', height=400)