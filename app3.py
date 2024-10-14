import os
import shutil
from flask import Flask, request, jsonify, render_template
from llama_index.core import (
    StorageContext, load_index_from_storage,
    VectorStoreIndex, SimpleDirectoryReader, ChatPromptTemplate
)
from llama_index.llms.huggingface import HuggingFaceInferenceAPI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Set your Hugging Face API key
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "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")

# Configure the Llama index settings
Settings.llm = HuggingFaceInferenceAPI(
    model_name="google/gemma-1.1-7b-it",
    tokenizer_name="google/gemma-1.1-7b-it",
    context_window=3000,
    token=HUGGINGFACE_API_KEY,
    max_new_tokens=512,
    generate_kwargs={"temperature": 0.1},
)

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Define the directory for persistent storage and data
PERSIST_DIR = "./db"
DATA_DIR = "data"

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PERSIST_DIR, exist_ok=True)

# Function to process PDF and index data
def data_ingestion():
    
    # Clear existing persistent data before processing new PDF
    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)  # Remove existing persistent storage
    os.makedirs(PERSIST_DIR, exist_ok=True)  # Recreate the directory for new storage

    # Load and index the new PDF
    documents = SimpleDirectoryReader(DATA_DIR).load_data()
    storage_context = StorageContext.from_defaults()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)

# Function to handle user query
def handle_query(query):
    # Load the stored index from the storage context
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
    
    chat_text_qa_msgs = [
        (
            "user",
            """You are a Q&A assistant named CHATTO, created by Suriya. You have a specific response programmed for when users specifically ask about your creator, Suriya. The response is: "I was created by Suriya, an enthusiast in Artificial Intelligence. He is dedicated to solving complex problems and delivering innovative solutions. With a strong focus on machine learning, deep learning, Python, generative AI, NLP, and computer vision, Suriya is passionate about pushing the boundaries of AI to explore new possibilities." For all other inquiries, your main goal is to provide answers as accurately as possible, based on the instructions and context you have been given. If a question does not match the provided context or is outside the scope of the document, kindly advise the user to ask questions within the context of the document.
            Context:
            {context_str}
            Question:
            {query_str}
            """
        )
    ]
    text_qa_template = ChatPromptTemplate.from_messages(chat_text_qa_msgs)
    
    query_engine = index.as_query_engine(text_qa_template=text_qa_template)
    answer = query_engine.query(query)
    
    if hasattr(answer, 'response'):
        formatted_answer = answer.response.replace("\n", "<br>")
        return formatted_answer
    elif isinstance(answer, dict) and 'response' in answer:
        formatted_answer = answer['response'].replace("\n", "<br><br>")
        return formatted_answer
    else:
        return "Sorry, I couldn't find an answer."

# Serve the static HTML file
@app.route('/')
def index():
    return render_template('index3.html')

# Upload PDF and process it
@app.route('/upload', methods=['POST'])
def upload_pdf():
    if 'pdf' not in request.files:
        return jsonify({"error": "No PDF file provided"}), 400
    
    pdf_file = request.files['pdf']
    file_path = os.path.join(DATA_DIR, pdf_file.filename)

    try:
        # Save the uploaded PDF
        pdf_file.save(file_path)

        # Clear previous data and process new PDF
        data_ingestion()

        return jsonify({"message": "PDF uploaded and processed successfully"})

    except Exception as e:
        return jsonify({"error": f"Failed to process PDF. {str(e)}"}), 500

# Handle query
@app.route('/query', methods=['POST'])
def query_document():
    data = request.get_json()
    query = data.get('query')
    
    if not query:
        return jsonify({"error": "No query provided"}), 400

    try:
        # Handle the user's query
        answer = handle_query(query)
        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"error": f"An error occurred while processing the query: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
