# RAG (Retrieval-Augmented Generation) System

This system implements a RAG-based question-answering system using LangChain, local LLM (via LM Studio), and Chroma vector store.

## Prerequisites

- Python 3.8+
- LM Studio running locally
- Required Python packages

## Installation

1. Clone the repository:
```

## System Components

- **LM Studio**: Hosts the local LLM model and provides an OpenAI-compatible API
- **LangChain**: Orchestrates the RAG pipeline
- **Chroma**: Vector store for document embeddings
- **Custom Retriever**: Enhances document retrieval with surrounding context

## Usage

1. Ensure LM Studio is running with a model loaded and server started
2. Run the query script
3. Enter your questions at the prompt
4. Type 'quit' to exit

## Notes

- The system uses LM Studio's local API (http://localhost:1234)
- Default model path is configured for Meta-Llama-3-8B-Instruct-GGUF
- Adjust the `max_tokens_limit` in the code based on your model's capabilities

## License

[Your License]