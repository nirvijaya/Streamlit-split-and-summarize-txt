import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Streamlit UI
st.set_page_config(page_title="AI Text Summarizer", layout="centered")

# Sidebar: OpenAI API Key Input
st.sidebar.header("🔑 Enter Your OpenAI API Key")
openai_api_key = st.sidebar.text_input("API Key", type="password")

# Main Section: Upload a File
st.title("📄 AI-Powered Text Summarizer")
st.write("Upload a text document and get an AI-generated summary.")

uploaded_file = st.file_uploader("Upload a text file", type=["txt"])

if openai_api_key and uploaded_file is not None:
    # Read file content
    file_text = uploaded_file.read().decode("utf-8")
    
    # Define a LangChain prompt template
    prompt_template = PromptTemplate(
        input_variables=["text"],
        template="Summarize the following text in a concise and clear manner:\n\n{text}"
    )

    # Initialize LangChain's OpenAI model
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-4", temperature=0.5)
    
    # Create a LangChain summary chain
    summary_chain = LLMChain(llm=llm, prompt=prompt_template)

    # Generate summary correctly
    with st.spinner("Generating summary..."):
        summary = summary_chain.invoke({"text": file_text})  # <-- Correct method to call LLMChain

    # Display the result
    st.subheader("🔍 Summary of the Document")
    st.write(summary["text"])  # Ensure correct key extraction
else:
    st.warning("Please enter your OpenAI API Key and upload a text file to proceed.")
