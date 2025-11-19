import os
import pickle
from typing import List, Dict, Any, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
import streamlit as st
from config import OPENAI_API_KEY, VECTOR_STORE_PATH

class VectorStoreManager:
    """Manages vector store operations for RAG system"""
    
    def __init__(self):
        if OPENAI_API_KEY:
            self.embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        else:
            self.embeddings = None
        self.vector_store = None
        self.documents = []
        self.metadata = {}
        
    def create_documents(self, texts: List[str], metadata: Dict[str, Any]) -> List[Document]:
        """Create Document objects from texts with metadata"""
        documents = []
        for i, text in enumerate(texts):
            doc_metadata = {
                "source": metadata.get("filename", "unknown"),
                "chunk_id": i,
                **metadata
            }
            documents.append(Document(page_content=text, metadata=doc_metadata))
        return documents
    
    def add_documents(self, texts: List[str], metadata: Dict[str, Any]):
        """Add documents to vector store"""
        if not self.embeddings:
            st.error("OpenAI API key not configured. Please set your API key in the .env file.")
            return
            
        try:
            new_documents = self.create_documents(texts, metadata)
            self.documents.extend(new_documents)
            
            if self.vector_store is None:
                # Create new vector store
                self.vector_store = FAISS.from_documents(new_documents, self.embeddings)
            else:
                # Add to existing vector store
                self.vector_store.add_documents(new_documents)
            
            # Update metadata registry with a unique identifier
            doc_id = metadata.get("document_id") or f"{metadata.get('filename', 'unknown')}_{len(self.metadata) + 1}"
            metadata["document_id"] = doc_id
            # Store a shallow copy to avoid Streamlit session mutations altering stored data
            self.metadata[doc_id] = metadata.copy()
            
            st.success(f"Successfully added {len(new_documents)} chunks to vector store")
            
        except Exception as e:
            st.error(f"Error adding documents to vector store: {str(e)}")
    
    def search_similar(
        self,
        query: str,
        k: int = 5,
        document_ids: Optional[List[str]] = None
    ) -> List[Document]:
        """Search for similar documents"""
        if not self.embeddings:
            st.error("OpenAI API key not configured. Please set your API key in the .env file.")
            return []
            
        try:
            if self.vector_store is None:
                return []
            
            working_docs = self.documents
            if document_ids:
                working_docs = [
                    doc for doc in self.documents
                    if doc.metadata.get("document_id") in document_ids
                ]
                if not working_docs:
                    return []
                temp_store = FAISS.from_documents(working_docs, self.embeddings)
                results = temp_store.similarity_search(query, k=k)
            else:
                results = self.vector_store.similarity_search(query, k=k)
            
            return results
            
        except Exception as e:
            st.error(f"Error searching vector store: {str(e)}")
            return []
    
    def save_vector_store(self, path: str = VECTOR_STORE_PATH):
        """Save vector store to disk"""
        try:
            os.makedirs(path, exist_ok=True)
            
            if self.vector_store:
                self.vector_store.save_local(path)
            
            # Save metadata
            with open(os.path.join(path, "metadata.pkl"), "wb") as f:
                pickle.dump(self.metadata, f)
            
            # Save documents
            with open(os.path.join(path, "documents.pkl"), "wb") as f:
                pickle.dump(self.documents, f)
                
        except Exception as e:
            st.error(f"Error saving vector store: {str(e)}")
    
    def load_vector_store(self, path: str = VECTOR_STORE_PATH):
        """Load vector store from disk"""
        try:
            if not os.path.exists(path):
                return False
            
            # Load vector store with security setting
            # Note: allow_dangerous_deserialization=True is safe here because we're loading
            # our own locally created files, not files from untrusted sources
            if os.path.exists(os.path.join(path, "index.faiss")):
                if self.embeddings:
                    self.vector_store = FAISS.load_local(
                        path, 
                        self.embeddings, 
                        allow_dangerous_deserialization=True
                    )
            
            # Load metadata
            metadata_path = os.path.join(path, "metadata.pkl")
            if os.path.exists(metadata_path):
                with open(metadata_path, "rb") as f:
                    self.metadata = pickle.load(f)
            
            # Load documents
            documents_path = os.path.join(path, "documents.pkl")
            if os.path.exists(documents_path):
                with open(documents_path, "rb") as f:
                    self.documents = pickle.load(f)
            
            return True
            
        except Exception as e:
            st.error(f"Error loading vector store: {str(e)}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        total_documents = len(self.metadata)
        return {
            "total_documents": total_documents,
            "total_chunks": len(self.documents),
            "has_vector_store": self.vector_store is not None
        }

    def get_document_library(self) -> List[Dict[str, Any]]:
        """Return a summarized view of uploaded documents"""
        library = []
        for meta in self.metadata.values():
            library.append({
                "Document": meta.get("display_name") or meta.get("filename", "Untitled"),
                "Tags": ", ".join(meta.get("tags", [])) if meta.get("tags") else "",
                "Source Type": meta.get("source_type", "PDF"),
                "Chunks": meta.get("chunk_count", 0),
                "Words": meta.get("word_count", 0),
                "Uploaded": meta.get("upload_date", "N/A"),
                "Document ID": meta.get("document_id")
            })
        # Sort by upload date descending when possible
        library.sort(key=lambda item: item.get("Uploaded", ""), reverse=True)
        return library

    def get_documents_by_ids(self, document_ids: List[str]) -> List[Document]:
        """Return raw Document chunks that belong to specific uploads"""
        if not document_ids:
            return []
        return [
            doc for doc in self.documents
            if doc.metadata.get("document_id") in document_ids
        ]
