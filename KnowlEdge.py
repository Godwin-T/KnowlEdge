import streamlit as st
from helperfunctions import (get_pdf_text, get_text_chunks,
                             get_vectorstore, get_conversation_chain)
from langchain_core.messages import AIMessage, HumanMessage
from HtmlTemplate import chat_css, user_template, bot_template

st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
st.markdown(chat_css, unsafe_allow_html=True)

def handle_userinput(user_question):
    """
    Handles user input during the chat interaction.

    Parameters:
    - user_question (str): The user's question or input.

    Performs the following actions:
    - Displays chat history messages in alternating user and bot templates.
    - Invokes the conversation chain to generate a response based on user input.
    - Appends the AI's response to the chat history.
    """
    # Display chat history messages
    for i, message in enumerate(st.session_state.chat_history[1:]):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

    # Invoke conversation chain with user input and chat history
    response = st.session_state.conversation.invoke({'input': user_question, "chat_history": st.session_state.chat_history})

    # Append AI's response to chat history
    st.session_state.chat_history.append(AIMessage(content=response['answer']))

    # Display AI's response
    st.write(bot_template.replace(
        "{{MSG}}", response['answer']), unsafe_allow_html=True)


def main():
    """
    Main function for the Streamlit application.

    Performs the following actions:
    - Initializes chat interface with HTML templates.
    - Manages session state for conversation and chat history.
    - Receives user input and updates chat history accordingly.
    - Processes uploaded PDF documents to initialize conversation chain.
    """
    st.write(chat_css, unsafe_allow_html=True)  # Display CSS for chat formatting
    st.header("Chat with multiple PDFs :books:")  # Display header for the chat application

    # Initialize session state variables if not already present
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = ['Nothing Yet']

    # Receive user input and add it to the chat history
    user_question = st.chat_input("Enter your message:", key="chat_input")
    if user_question:
        st.session_state.chat_history.append(HumanMessage(content=user_question))
        handle_userinput(user_question)

    # Sidebar section for uploading PDF documents
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # Get text from PDFs
                raw_text = get_pdf_text(pdf_docs)

                # Split text into manageable chunks
                text_chunks = get_text_chunks(raw_text)

                # Create vector store from text chunks
                vectorstore = get_vectorstore(text_chunks)

                # Create conversation chain based on vector store
                st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == '__main__':
    main()
