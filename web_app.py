from flask import Flask, render_template, request, jsonify, Response
from query_documents import initialize_rag
import logging
import json

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

@app.route('/stream', methods=['POST'])
def stream():
    if not qa_chain:
        return jsonify({'error': 'RAG system not initialized'}), 500

    try:
        question = request.json.get('question', '').strip()
        if not question:
            return jsonify({'error': 'Please enter a valid question'}), 400

        def generate():
            # Send the starting message
            yield 'data: {"type": "start"}\n\n'

            # Process the question with streaming
            result = qa_chain({
                "question": question,
                "chat_history": chat_history
            })

            # Stream the answer word by word
            words = result["answer"].split()
            for i, word in enumerate(words):
                data = {
                    "type": "token",
                    "content": word + (" " if i < len(words) - 1 else "")
                }
                yield f'data: {json.dumps(data)}\n\n'

            # Send sources at the end
            sources = []
            if result.get("source_documents"):
                sources = [doc.metadata.get('source', 'Unknown source') 
                          for doc in result["source_documents"]]
            
            data = {
                "type": "end",
                "sources": sources
            }
            yield f'data: {json.dumps(data)}\n\n'

            # Add to chat history after completion
            chat_history.append((question, result["answer"]))

        return Response(generate(), mimetype='text/event-stream')

    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        error_data = {
            "type": "error",
            "content": "An error occurred processing your question"
        }
        return f'data: {json.dumps(error_data)}\n\n'

if __name__ == '__main__':
    app.run(debug=True, port=5000) 