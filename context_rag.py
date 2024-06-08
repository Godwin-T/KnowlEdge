import os
import base64
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from HtmlTemplate import chat_css, user_template, bot_template
from langchain_core.messages import AIMessage, HumanMessage

os.environ["OPENAI_API_KEY"] = "REMOVED_SECRET"
# Streamlit application
st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
st.markdown(chat_css, unsafe_allow_html=True)


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks
def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorestore):

    contextualization_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""


    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Use three sentences maximum and keep the answer concise.\

    {context}"""

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

    llm = ChatOpenAI()
    retriever = vectorestore.as_retriever()
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain

def model_generation(user_question, chat_history, rag_chain):

    response = rag_chain.invoke({"input": user_question, "chat_history":chat_history})
    chat_history.extend([HumanMessage(content=user_question), AIMessage(content =response["answer"])])

# def stream_data(data):
#     for word in data.split(" "):
#         yield word + " "


def handle_userinput(user_question):

    for i, message in enumerate(st.session_state.chat_history[1:]):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html= True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html= True)

    response = st.session_state.conversation.invoke({'input': user_question, "chat_history": st.session_state.chat_history})['answer']
    st.session_state.chat_history.append(AIMessage(content =response))
    st.write(bot_template.replace(
                "{{MSG}}", response), unsafe_allow_html= True)



def main():

    st.write(chat_css, unsafe_allow_html=True)
    st.header("Chat with multiple PDFs :books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = ['Nothing Yet']

    # Receive user input and add it to the chat history
    user_question = st.chat_input("Enter your message:", key="chat_input")
    if user_question:
        st.session_state.chat_history.append(HumanMessage(content=user_question))
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)


if __name__ == '__main__':
    main()
