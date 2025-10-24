import os
from dotenv import load_dotenv
from langchain_community.llms import Ollama
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables from .env file
load_dotenv()

# Set environment variables for LangChain
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')

# Create the chat prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that helps people find information."),
    ("user", "Question: {question}")
])

# Streamlit UI
st.title("LangChain Demo with Gemma 3:1B Model")
input_text = st.text_input("Enter your question here")

# Initialize the Ollama model
llm = Ollama(model="gemma3:1b", temperature=0)
output_parser = StrOutputParser()

# Combine components into a chain
chain = prompt | llm | output_parser

# Handle user input
if input_text:
    response = chain.invoke({"question": input_text})
    st.write(response)
