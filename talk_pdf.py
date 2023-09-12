import streamlit as st
from PyPDF2 import PdfReader
from streamlit_tags import st_tags
import pandas as pd
import base64
import spacy
import en_core_web_sm

from PyPDF2 import PdfReader
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
import time
import PyPDF2
import openai
import ast
import os
from dotenv import load_dotenv
from langchain import PromptTemplate, LLMChain
from langchain.chains import SequentialChain
from langchain.chat_models import ChatOpenAI
from langchain.chat_models import AzureChatOpenAI

from functions import pdf_to_text,write_text_file

hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}

       .st-b7 {
    color: white;
    font-size=10px;
}

       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)






st.title('CHAT WITH YOUR PDF')

uploaded_file = st.file_uploader("Upload your ' PDF ' DOCUMENT", type="pdf")



with st.expander("WHAT IS THIS ?"):
    st.write('This is an intelligent document Q A  made by HELMI MASSOUSSI')


with st.expander("HOW IT IS WORK ?"):
    st.write('Just drag & drop the document and start asking')

if uploaded_file:
    load_dotenv()
    

    
    st.header('ASK YOUR DOCUMENT: ')
    file = uploaded_file.name
    txt=pdf_to_text(file)
    path='doc.txt'
    write_text_file(txt, path)  

    loader = TextLoader(path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()

    docsearch = Chroma.from_documents(texts, embeddings)

    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever(search_kwargs={"k": 1}))

     
    st.success("File Loaded Successfully!!")
    
    #question = st.text_input("Ask something from the file", placeholder="Find something similar to: ....this.... in the text?", disabled=not uploaded_file,)    
    #if question:

    # Initialize chat history
    if "messages" not in st.session_state:
      st.session_state.messages = []

# Display chat messages from history on app rerun
    for message in st.session_state.messages:
      with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
    if prompt := st.chat_input("Ask a question"):
      
    # Add user message to chat history
      st.session_state.messages.append({"role": "user", "content": prompt})
      with st.chat_message("user"):
        st.markdown(prompt)

      
    # Display assistant response in chat message container
      with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        assistant_response = qa.run(str(prompt))

        for chunk in assistant_response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)




    # Add assistant response to chat history
      st.session_state.messages.append({"role": "assistant", "content": full_response})
      




     

        #response = qa.run(str(question))       
        #st.write(response)
    
    

    

    
        
    

    

