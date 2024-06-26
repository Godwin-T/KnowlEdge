import os
from PyPDF2 import PdfReader  # Import PDF reader
from langchain_community.vectorstores import FAISS  # Import vector store
from langchain_groq.chat_models import ChatGroq  # Import ChatGroq model
from langchain_community.embeddings import OpenAIEmbeddings # Import OpenAI embeddings
from langchain.text_splitter import CharacterTextSplitter  # Import CharacterTextSplitter
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain  # Import function for creating a document processing chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain  # Import functions for creating retrieval chains


def get_pdf_text(pdf_docs):
    """
    Extracts text from a list of PDF documents.

    Parameters:
    - pdf_docs (list): List of paths to PDF documents.

    Returns:
    - text (str): Extracted text from all provided PDF documents.
    """
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    """
    Splits the provided text into manageable chunks.

    Parameters:
    - text (str): The text to be split into chunks.

    Returns:
    - chunks (list): List of text chunks.
    """
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    """
    Creates a vector store from text chunks using embeddings.

    Parameters:
    - text_chunks (list): List of text chunks.

    Returns:
    - vectorstore (FAISS): Vector store created from the provided text chunks.
    """
    embeddings = OpenAIEmbeddings()  # Initialize OpenAI embeddings
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)  # Create vector store using FAISS
    return vectorstore


def get_conversation_chain(vectorstore):
    """
    Sets up a conversational chain for question-answering tasks.

    Parameters:
    - vectorstore (FAISS): Vector store for text chunks.

    Returns:
    - rag_chain: A retrieval-augmented generation (RAG) chain configured for question-answering.
    """
    # Define prompts for contextualization and question-answering
    contextualization_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""

    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Use three sentences maximum and keep the answer concise.\

    {context}"""

    # Define chat prompts templates
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualization_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    llm = ChatGroq()  # Initialize ChatGroq model
    retriever = vectorstore.as_retriever()  # Convert vector store to a retriever
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )  # Create history-aware retriever for context-aware question-answering
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)  # Create chain for question-answering
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)  # Combine retriever and question-answering chains

    return rag_chain  # Return the configured retrieval-augmented generation chain
