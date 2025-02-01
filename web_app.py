from flask import Flask, render_template, request, jsonify
from query_documents import initialize_rag
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize the RAG system once when the app starts
try:
    qa_chain = initialize_rag()
    chat_history = []
    logger.info("RAG System successfully initialized for web interface")
except Exception as e:
    logger.error(f"Failed to initialize RAG system: {str(e)}")
    qa_chain = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    if not qa_chain:
        return jsonify({'error': 'RAG system not initialized'}), 500

    try:
        question = request.json.get('question', '').strip()
        if not question:
            return jsonify({'error': 'Please enter a valid question'}), 400

        # Process the question
        result = qa_chain({
            "question": question,
            "chat_history": chat_history
        })

        # Format sources
        sources = []
        if result.get("source_documents"):
            sources = [doc.metadata.get('source', 'Unknown source') 
                      for doc in result["source_documents"]]

        # Add to chat history
        chat_history.append((question, result["answer"]))

        return jsonify({
            'answer': result["answer"],
            'sources': sources
        })

    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        return jsonify({'error': 'An error occurred processing your question'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) 