from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI


def main():
    load_dotenv()
    st.set_page_config(page_title="Query Your PDF documnet💬")
    st.header("Query Your PDF documnet")

    #upload you file. 
    pdf = st.file_uploader("Upload your PDF file", type=["pdf"])

    #extract text from your pdf
    if pdf is not None:     
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
         text += page.extract_text()
                    
    #split into chunck
        text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=75,
        length_function=len            
        )
            
        chunks = text_splitter.split_text(text)

        embeddings = OpenAIEmbeddings()
        document = FAISS.from_texts(chunks, embeddings)  


        user_question = st.text_input("Enter your query here")     
        
       # st.write(chunks)
                              
            
        #st.write(text)


if __name__ == "__main__":
    main()
