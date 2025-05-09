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
        embeddings = CohereEmbeddings(cohere_api_key=os.getenv("COHERE_API_KEY"))
        vectordb = Chroma(persist_directory='db', embedding_function=embeddings)
        
        qa = RetrievalQA.from_chain_type(
            llm=Cohere(cohere_api_key=os.getenv("COHERE_API_KEY")),
            chain_type="refine",
            retriever=vectordb.as_retriever(),
            return_source_documents=True
        )
        return qa
    except Exception as e:
        return None

# Load database on startup
qa = load_db()

# Knowledge Base Answer Function
def answer_from_knowledgebase(message):
    if not qa:
        return "Knowledge base not loaded. Please check configuration."
    
    try:
        result = qa({"query": message})
        
        if not result or 'result' not in result:
            return "No answer found in the knowledge base."
        
        return result['result']
    except Exception as e:
        return f"Error retrieving answer: {e}"

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