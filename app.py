import os
import base64
import streamlit as st
from HtmlTemplate import chat_css, user_template, bot_template
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

os.environ["OPENAI_API_KEY"] = "REMOVED_SECRET"
# Streamlit application
st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
st.markdown(chat_css, unsafe_allow_html=True)


# File uploader to upload a PDF file
# with st.sidebar:
#     uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

#     if uploaded_file is not None:
#         # Read the PDF file as bytes
#         pdf_bytes = uploaded_file.read()

#         # Base64 encode the PDF bytes
#         base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')

#         # Embed PDF in the page using an iframe
#         pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="600" height="1000" type="application/pdf"></iframe>'
#         st.markdown(pdf_display, unsafe_allow_html=True)

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
def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    prompt = ChatPromptTemplate("You are a helpful reading assistant and you are to answer based on the retrieval information provided only")
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        stream=True
    )
    return conversation_chain

def handle_userinput(user_question):

    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    print(st.session_state.chat_history)

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)




#st.write('<div style="height: 650px;"></div>', unsafe_allow_html=True)

def main():
   # load_dotenv()
    # st.set_page_config(page_title="Chat with multiple PDFs",
    #                    page_icon=":books:")
    st.write(chat_css, unsafe_allow_html=True)
    st.header("Chat with multiple PDFs :books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # Receive user input and add it to the chat history
    user_question = st.chat_input("Enter your message:", key="chat_input")
    if user_question:
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


# def chat_mode(model):
#     # Function for interactive chat mode with a language model
#     chat_history = ChatMessageHistory()

#     # Initialize the chat messages in the session state
#     if "messages" not in st.session_state:
#         st.session_state.messages = []

#
#     # Receive user input and add it to the chat history
#     user_input = st.chat_input("Enter your message:", key="chat_input")
#     if user_input:
#         chat_history.add_user_message(user_input)
#         st.chat_message("user").markdown(user_input)
#         st.session_state.messages.append({"role": "user", "content": user_input})




# def pdf_embedding():

# def emb_storage():

# def rag_pipeline():

# def agent_pipeline():

# def youtube_use_links():

# def youtube_tutorial_links():
