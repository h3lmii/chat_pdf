import streamlit as st
from PyPDF2 import PdfReader
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma



def pdf_to_text(path,nlp_form=False):
  '''
  convert pdf file to text string and nlp for spacy if it's True
  '''
  txt=''
  cv=PdfReader(path)
  for page in cv.pages:
    txt+=page.extract_text()
  if nlp_form:
    return nlp(txt)
  else:
    return txt



def write_text_file(content, file_path):
    try:
        with open(file_path, 'w') as file:
            file.write(content)
        return True
    except Exception as e:
        print(f"Error occurred while writing the file: {e}")
        return False
