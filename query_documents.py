from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from openai import OpenAI
import os
import logging
from dotenv import load_dotenv
from embed_documents import LMStudioEmbeddings
from typing import List
from langchain.docstore.document import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure OpenAI client for LM Studio
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

class ExpandedRetriever(BaseRetriever):
    """Custom retriever that expands context with surrounding chunks."""
    
    vectorstore: Chroma = Field(description="Vector store for document retrieval")
    base_retriever: BaseRetriever = Field(description="Base retriever for initial document fetch")
    window: int = Field(default=3, description="Window size for context expansion")
    
    class Config:
        arbitrary_types_allowed = True
    
    @classmethod
    def from_components(
        cls,
        vectorstore: Chroma,
        base_retriever: BaseRetriever,
        window: int = 3
    ) -> "ExpandedRetriever":
        """Create an ExpandedRetriever from components."""
        return cls(
            vectorstore=vectorstore,
            base_retriever=base_retriever,
            window=window
        )
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        # Get base results
        docs = self.base_retriever.get_relevant_documents(query)
        # Expand with surrounding context
        return expand_retrieved_documents(self.vectorstore, docs, self.window)
        
    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        # Async implementation if needed
        return self._get_relevant_documents(query)

def get_surrounding_chunks(vectorstore: Chroma, doc: Document, window: int = 3) -> List[Document]:
    """Get surrounding chunks for a given document."""
    try:
        # Get the document's source and index
        source = doc.metadata.get('source')
        chunk_idx = doc.metadata.get('chunk_idx')
        
        if source is None or chunk_idx is None:
            return []
        
        # Query for all chunks from the same source
        all_chunks = vectorstore.get(
            where={"source": source},
            include=['documents', 'metadatas']
        )
        
        if not all_chunks:
            return []
            
        # Sort chunks by their index
        sorted_chunks = sorted(
            zip(all_chunks['documents'], all_chunks['metadatas']),
            key=lambda x: x[1].get('chunk_idx', 0)
        )
        
        # Find the position of our target chunk
        target_pos = next(
            (i for i, (_, meta) in enumerate(sorted_chunks) 
             if meta.get('chunk_idx') == chunk_idx),
            None
        )
        
        if target_pos is None:
            return []
            
        # Get window chunks before and after
        start_idx = max(0, target_pos - window)
        end_idx = min(len(sorted_chunks), target_pos + window + 1)
        
        # Create Document objects for the surrounding chunks
        context_chunks = []
        for i in range(start_idx, end_idx):
            text, metadata = sorted_chunks[i]
            context_chunks.append(Document(
                page_content=text,
                metadata=metadata
            ))
        
        return context_chunks
        
    except Exception as e:
        logger.error(f"Error getting surrounding chunks: {str(e)}")
        return []

def expand_retrieved_documents(vectorstore: Chroma, docs: List[Document], window: int = 3) -> List[Document]:
    """Expand the retrieved documents with surrounding context."""
    # Use a dictionary with tuple of source and chunk_idx as key to avoid duplicates
    expanded_docs_dict = {}
    
    for doc in docs:
        # Create a key from source and chunk_idx
        key = (doc.metadata.get('source', ''), doc.metadata.get('chunk_idx', 0))
        expanded_docs_dict[key] = doc
        
        # Get and add surrounding chunks
        surrounding_chunks = get_surrounding_chunks(vectorstore, doc, window)
        for chunk in surrounding_chunks:
            chunk_key = (chunk.metadata.get('source', ''), chunk.metadata.get('chunk_idx', 0))
            expanded_docs_dict[chunk_key] = chunk
    
    # Convert back to list and sort by source and chunk index
    sorted_docs = sorted(
        expanded_docs_dict.values(),
        key=lambda x: (
            x.metadata.get('source', ''),
            x.metadata.get('chunk_idx', 0)
        )
    )
    
    return sorted_docs

def initialize_rag():
    try:
        # Check if database exists
        if not os.path.exists("./db"):
            logger.error("Database directory './db' not found. Please run embed_documents.py first.")
            raise FileNotFoundError("Database not found")

        # Initialize embeddings using our custom LM Studio embeddings class
        logger.info("Initializing LM Studio embeddings...")
        embeddings = LMStudioEmbeddings()

        # Check LM Studio connection
        try:
            response = client.models.list()
            logger.info("Successfully connected to LM Studio")
        except Exception as e:
            logger.error(f"Failed to connect to LM Studio: {str(e)}")
            logger.error("Please make sure LM Studio is running on http://localhost:1234")
            raise

        # Load the persisted Chroma database
        logger.info("Loading Chroma database...")
        try:
            vectorstore = Chroma(
                persist_directory="./db",
                embedding_function=embeddings
            )
            logger.info("Successfully loaded Chroma database")
        except Exception as e:
            logger.error(f"Failed to load Chroma database: {str(e)}")
            raise

        # Initialize the LLM with LM Studio settings
        logger.info("Initializing LLM...")
        try:
            llm = ChatOpenAI(
                model="LM Studio Community/Meta-Llama-3-8B-Instruct-GGUF",
                temperature=0.7,
                max_tokens=-1,
                openai_api_base="http://localhost:1234/v1",
                openai_api_key="lm-studio",
                streaming=True,
            )
            logger.info("Successfully initialized LLM")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {str(e)}")
            raise

        # Create a custom retriever that expands context
        logger.info("Setting up retriever...")
        try:
            base_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
            expanded_retriever = ExpandedRetriever.from_components(
                vectorstore=vectorstore,
                base_retriever=base_retriever,
                window=3
            )
            logger.info("Successfully set up retriever")
        except Exception as e:
            logger.error(f"Failed to set up retriever: {str(e)}")
            raise

        # Create the conversational chain with expanded retriever
        logger.info("Creating conversational chain...")
        try:
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=expanded_retriever,
                return_source_documents=True,
                verbose=True,
                # Add these new parameters for better chain configuration
                chain_type="stuff",  # Use 'stuff' method to combine documents
                rephrase_question=True,  # Allow question rephrasing for better context
                max_tokens_limit=4000,  # Adjust based on your model's context window
            )
            logger.info("Successfully created conversational chain")
        except Exception as e:
            logger.error(f"Failed to create conversational chain: {str(e)}")
            raise
        
        return qa_chain
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {str(e)}")
        raise

def main():
    try:
        qa_chain = initialize_rag()
        logger.info("RAG System successfully initialized")
        chat_history = []
        
        print("RAG System initialized. Type 'quit' to exit.")
        
        while True:
            question = input("\nYour question: ").strip()
            if question.lower() == 'quit':
                break
            if not question:
                print("Please enter a valid question.")
                continue
                
            try:
                result = qa_chain({
                    "question": question, 
                    "chat_history": chat_history
                })
                print("\nAnswer:", result["answer"])
                
                if result["source_documents"]:
                    print("\nSources:")
                    for doc in result["source_documents"]:
                        print(f"- {doc.metadata.get('source', 'Unknown source')}")
                else:
                    print("\nNo sources found for this answer.")
                
                chat_history.append((question, result["answer"]))
                
            except Exception as e:
                logger.error(f"Error processing question: {str(e)}")
                print("An error occurred while processing your question. Please try again.")
                
    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}")
        print("Failed to initialize the RAG system. Please check your configuration.")

if __name__ == "__main__":
    main() 