import os
from flask import Flask, render_template, request, jsonify, abort

# LangChain and AI Imports
from langchain.llms import Cohere
from langchain.embeddings import CohereEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA, ConversationChain
from langchain.memory import ConversationBufferMemory

# Environment and Configuration
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Flask Application Setup
app = Flask(__name__)

# Initialize conversation memory
memory = ConversationBufferMemory()

# Database Loading Function
def load_db():
    try:
        print("Starting database load...")
        embeddings = CohereEmbeddings(cohere_api_key=os.getenv("COHERE_API_KEY"))
        
        # Update path to match your database location
        vectordb = Chroma(persist_directory='db/index', embedding_function=embeddings)
        
        # Check the number of documents
        doc_count = vectordb._collection.count()
        print(f"Total documents in collection: {doc_count}")
        
        if doc_count == 0:
            print("Warning: The vector database is empty.")
            return None
            
        print("Initializing RetrievalQA chain...")
        qa = RetrievalQA.from_chain_type(
            llm=Cohere(cohere_api_key=os.getenv("COHERE_API_KEY")),
            chain_type="refine",
            retriever=vectordb.as_retriever(),
            return_source_documents=True,
            verbose=True  # Enable verbose mode for debugging
        )
        print("RetrievalQA chain initialized successfully")
        return qa
        
    except Exception as e:
        print(f"Database Loading Error: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return None

# Load database on startup
qa = load_db()

# Knowledge Base Answer Function
def answer_from_knowledgebase(message):
    if not qa:
        print("QA system not initialized")
        return "Knowledge base not loaded. Please check configuration."
    
    try:
        print(f"Querying knowledge base with: {message}")
        result = qa({"query": message})
        print(f"Raw result: {result}")  # Debug print
        
        if not result:
            return "No result returned from knowledge base."
            
        if isinstance(result, dict):
            if 'result' in result:
                return result['result']
            else:
                print(f"Available keys in result: {result.keys()}")
                return "Result format unexpected. Check logs for details."
        else:
            print(f"Unexpected result type: {type(result)}")
            return "Unexpected response format from knowledge base."
            
    except Exception as e:
        print(f"Error details: {str(e)}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return f"Error retrieving answer: {str(e)}"

# Knowledge Base Search Function
def search_knowledgebase(message):
    if not qa:
        return "Knowledge base not loaded. Please check configuration."
    
    try:
        result = qa({"query": message})
        
        sources = ""
        for count, doc in enumerate(result.get('source_documents', []), 1):
            sources += f"Source {count}:\n{doc.page_content}\n\n"
        
        return sources
    except Exception as e:
        return f"Error searching knowledge base: {e}"

# Chatbot Answer Function
def answer_as_chatbot(message):
    try:
        llm = Cohere(
            cohere_api_key=os.getenv("COHERE_API_KEY"),
            model="command"
        )

        conversation = ConversationChain(
            llm=llm, 
            memory=memory,
            verbose=True
        )

        response = conversation.predict(input=message)
        
        return response

    except Exception as e:
        return "I'm having trouble generating a response right now."

# Route Handlers
@app.route('/kbanswer', methods=['POST'])
def kbanswer():
    message = request.json['message']
    response = answer_from_knowledgebase(message)
    return jsonify({'message': response}), 200

@app.route('/search', methods=['POST'])
def search():
    message = request.json['message']
    sources = search_knowledgebase(message)
    return jsonify({'message': sources}), 200

@app.route('/answer', methods=['POST'])
def answer():
    message = request.json['message']
    response_message = answer_as_chatbot(message)
    return jsonify({'message': response_message}), 200

@app.route("/")
def index():
    return render_template("index.html", title="")

# Application Entry Point
if __name__ == "__main__":
    app.run(debug=True)