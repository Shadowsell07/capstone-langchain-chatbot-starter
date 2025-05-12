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
        
        # Verify database files exist
        if not (os.path.exists('db/chroma-collections.parquet') and 
                os.path.exists('db/chroma-embeddings.parquet')):
            print("Error: Required database files not found!")
            return None
            
        vectordb = Chroma(persist_directory='db', embedding_function=embeddings)
        
        # Verify document count
        doc_count = vectordb._collection.count()
        print(f"Found {doc_count} documents in database")
        
        qa = RetrievalQA.from_chain_type(
            llm=Cohere(cohere_api_key=os.getenv("COHERE_API_KEY")),
            chain_type="refine",
            retriever=vectordb.as_retriever(),
            return_source_documents=True
        )
        print("Database loaded successfully!")
        return qa
    except Exception as e:
        print(f"Error loading database: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return None

# Load database on startup
qa = load_db()

# Knowledge Base Answer Function
def answer_from_knowledgebase(message):
    if not qa:
        print("QA system not initialized")
        return "Knowledge base not loaded. Please check configuration."
    
    try:
        # First check if we can get any documents
        docs = qa.retriever.get_relevant_documents(message)
        print(f"Found {len(docs)} relevant documents")
        
        if not docs:
            return "I don't have any information about that in my knowledge base. Try asking about Wu-Tang Clan or use the regular chatbot mode for general questions."
        
        # If we have documents, get the answer
        res = qa({"query": message})
        
        if not res or 'result' not in res or not res['result'].strip():
            return "I found some related information but couldn't generate a specific answer. Try rephrasing your question or use the regular chatbot mode."
            
        return res['result']
    except Exception as e:
        print(f"Error in answer_from_knowledgebase: {str(e)}")
        return "I encountered an error while searching the knowledge base. Try using the regular chatbot mode instead."

# Knowledge Base Search Function
def search_knowledgebase(message):
    if not qa:
        print("QA system not initialized")
        return "Knowledge base not loaded. Please check configuration."
    
    try:
        # Use the RetrievalQA chain directly like in answer_from_knowledgebase
        res = qa({"query": message})
        
        if not res or 'result' not in res or not res['result'].strip():
            return "I found some related information but couldn't generate a specific answer. Try rephrasing your question or use the regular chatbot mode."
            
        return res['result']
    except Exception as e:
        print(f"Search error: {str(e)}")
        return "I encountered an error while searching the knowledge base. Try using the regular chatbot mode instead."

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

@app.route('/test-db', methods=['GET'])
def test_db():
    try:
        embeddings = CohereEmbeddings(cohere_api_key=os.getenv("COHERE_API_KEY"))
        vectordb = Chroma(persist_directory='db', embedding_function=embeddings)
        count = vectordb._collection.count()
        return jsonify({
            'status': 'success',
            'document_count': count
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        })

@app.route('/verify-db', methods=['GET'])
def verify_db():
    try:
        if not qa:
            return jsonify({
                'status': 'error',
                'message': 'QA system not initialized'
            })
        
        # Test retrieval
        test_docs = qa.retriever.get_relevant_documents("python")
        
        return jsonify({
            'status': 'success',
            'document_count': len(test_docs),
            'sample_content': test_docs[0].page_content[:200] if test_docs else None
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/dbstatus')
def db_status():
    if not qa:
        return jsonify({
            'status': 'error',
            'message': 'Database not loaded'
        })
    
    try:
        # Test the retriever
        test_docs = qa.retriever.get_relevant_documents("test query")
        return jsonify({
            'status': 'success',
            'documents_found': len(test_docs),
            'sample_query': "test query"
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route("/")
def index():
    return render_template("index.html", title="")

# Application Entry Point
if __name__ == "__main__":
    app.run(debug=True)