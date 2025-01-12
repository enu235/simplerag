from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import UnstructuredPDFLoader, PDFMinerLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from openai import OpenAI
import os
import json
import logging
from typing import List
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
import tiktoken

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure OpenAI client for LM Studio
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

class LMStudioEmbeddings(OpenAIEmbeddings):
    def __init__(self):
        super().__init__(openai_api_base="http://localhost:1234/v1", openai_api_key="lm-studio")
        self.client = client
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using LM Studio's endpoint"""
        embeddings = []
        for text in texts:
            text = text.replace("\n", " ")
            embedding = self.client.embeddings.create(
                input=[text], 
                model="LM Studio Community/Meta-Llama-3-8B-Instruct-GGUF"
            ).data[0].embedding
            embeddings.append(embedding)
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a query using LM Studio's endpoint"""
        text = text.replace("\n", " ")
        return self.client.embeddings.create(
            input=[text], 
            model="LM Studio Community/Meta-Llama-3-8B-Instruct-GGUF"
        ).data[0].embedding

def load_documents_from_directory(directory: str) -> List[Document]:
    """Load documents from a directory with better error handling"""
    documents = []
    
    # Define loaders with their respective file patterns
    loaders = [
        (DirectoryLoader(
            directory,
            glob="**/*.txt",
            show_progress=True
        ), "text files"),
        (DirectoryLoader(
            directory,
            glob="**/*.pdf",
            loader_cls=PDFMinerLoader,  # Changed to PDFMinerLoader for better reliability
            show_progress=True
        ), "PDF files")
    ]
    
    # Try loading each document type
    for loader, doc_type in loaders:
        try:
            logger.info(f"Loading {doc_type}...")
            docs = loader.load()
            documents.extend(docs)
            logger.info(f"Successfully loaded {len(docs)} {doc_type}")
        except Exception as e:
            logger.error(f"Error loading {doc_type}: {str(e)}")
            continue
    
    return documents

def load_web_articles(urls_file='web_sources.json') -> List[Document]:
    """Load articles from web sources with better error handling"""
    try:
        if not os.path.exists(urls_file):
            logger.warning(f"No {urls_file} found. Skipping web articles.")
            return []
            
        with open(urls_file, 'r') as f:
            urls = json.load(f)
        
        if not urls:
            logger.warning("No URLs found in the file.")
            return []
            
        web_loader = WebBaseLoader(urls)
        web_documents = web_loader.load()
        logger.info(f"Successfully loaded {len(web_documents)} web articles")
        return web_documents
        
    except json.JSONDecodeError:
        logger.error(f"Error parsing {urls_file}. Make sure it's valid JSON.")
        return []
    except Exception as e:
        logger.error(f"Error loading web articles: {str(e)}")
        return []

def load_and_embed_documents():
    try:
        # Create docs directory if it doesn't exist
        if not os.path.exists('./docs'):
            os.makedirs('./docs')
            logger.info("Created docs directory")
        
        # Load all documents
        documents = load_documents_from_directory('./docs')
        #web_documents = load_web_articles()
        
        # Combine all documents
        all_documents = documents # + web_documents
        
        if not all_documents:
            logger.warning("No documents were loaded. Please add documents to the docs folder or check web sources.")
            return
            
        logger.info(f"Total documents loaded: {len(all_documents)}")

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            is_separator_regex=False,
            keep_separator=True,
            strip_whitespace=True,
            add_start_index=True
        )
        
        splits = text_splitter.split_documents(documents)
        
        # Add chunk indices to metadata
        for i, doc in enumerate(splits):
            doc.metadata['chunk_idx'] = i
            if 'source' not in doc.metadata:
                doc.metadata['source'] = f'document_{i//10}'  # Group every 10 chunks
                
        logger.info(f"Split into {len(splits)} chunks")

        # Initialize embeddings using custom LM Studio embeddings class
        embeddings = LMStudioEmbeddings()

        # Create and persist Chroma vector store
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory="./db"
        )
        vectorstore.persist()
        logger.info("Documents successfully embedded and stored in Chroma DB")

    except Exception as e:
        logger.error(f"Error in load_and_embed_documents: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        load_and_embed_documents()
    except Exception as e:
        logger.error(f"Failed to process documents: {str(e)}")
        exit(1) 