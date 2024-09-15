import os
import streamlit as st
from beyondllm import source, retrieve, embeddings, llms, generator
import nltk

# Define a custom NLTK data directory within your app's local directory
nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

# Ensure necessary NLTK resources are available
nltk.download("punkt", download_dir=nltk_data_dir)

# Streamlit app title
st.title("Chat with Document")

# API Key Input
st.text("Enter API Key")
system_prompt = "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise."

# Secure input for API key
api_key = st.text_input("Google API Key:", type="password")
if api_key:
    os.environ['GOOGLE_API_KEY'] = api_key  # Store API key in the environment
    st.success("API Key entered successfully!")
else:
    st.warning("Please enter a valid API key to proceed.")

# File upload functionality
uploaded_file = st.file_uploader("Choose a PDF file", type='pdf')

# Text input for user's question
question = st.text_input("Enter your question")

# Execute when both file and question are provided
if uploaded_file is not None and question:

    # Save uploaded file locally in Streamlit session state
    save_path = "./uploaded_files"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    file_path = os.path.join(save_path, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Process the uploaded PDF
    data = source.fit(file_path, dtype="pdf", chunk_size=1024, chunk_overlap=50)

    # Create a retriever with the processed data
    retriever = retrieve.auto_retriever(data, type="normal", top_k=5)

    # Generate a response based on the question and retrieved context
    pipeline = generator.Generate(system_prompt=system_prompt, question=question, retriever=retriever)
    response = pipeline.call()

    # Display the response in Streamlit
    st.write(response)
else:
    if not uploaded_file:
        st.info("Please upload a PDF file.")
    if not question:
        st.info("Please enter a question.")

# Instructions as a caption
st.caption("Upload a PDF document and enter a question to query information from the document.")

