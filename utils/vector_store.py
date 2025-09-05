import os
import pickle
from typing import List, Dict, Any
import faiss
import numpy as np
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
import streamlit as st
from config import OPENAI_API_KEY, VECTOR_STORE_PATH, CHUNK_SIZE, CHUNK_OVERLAP

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
                "subject": metadata.get("subject", "general"),
                "topic": metadata.get("topic", "general"),
                "chapter": metadata.get("chapter", "general"),
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
            
            # Update metadata
            doc_id = f"{metadata.get('filename', 'unknown')}_{metadata.get('subject', 'general')}"
            self.metadata[doc_id] = metadata
            
            st.success(f"Successfully added {len(new_documents)} chunks to vector store")
            
        except Exception as e:
            st.error(f"Error adding documents to vector store: {str(e)}")
    
    def search_similar(self, query: str, k: int = 5, filter_dict: Dict = None) -> List[Document]:
        """Search for similar documents"""
        if not self.embeddings:
            st.error("OpenAI API key not configured. Please set your API key in the .env file.")
            return []
            
        try:
            if self.vector_store is None:
                return []
            
            if filter_dict:
                # Filter documents by metadata
                filtered_docs = []
                for doc in self.documents:
                    match = True
                    for key, value in filter_dict.items():
                        if doc.metadata.get(key) != value:
                            match = False
                            break
                    if match:
                        filtered_docs.append(doc)
                
                if not filtered_docs:
                    return []
                
                # Create temporary vector store for filtered docs
                temp_store = FAISS.from_documents(filtered_docs, self.embeddings)
                results = temp_store.similarity_search(query, k=k)
            else:
                results = self.vector_store.similarity_search(query, k=k)
            
            return results
            
        except Exception as e:
            st.error(f"Error searching vector store: {str(e)}")
            return []
    
    def get_all_subjects(self) -> List[str]:
        """Get all unique subjects in the vector store"""
        subjects = set()
        for doc in self.documents:
            subjects.add(doc.metadata.get("subject", "general"))
        return sorted(list(subjects))
    
    def get_topics_by_subject(self, subject: str) -> List[str]:
        """Get all topics for a specific subject"""
        topics = set()
        for doc in self.documents:
            if doc.metadata.get("subject") == subject:
                topics.add(doc.metadata.get("topic", "general"))
        return sorted(list(topics))
    
    def get_chapters_by_topic(self, subject: str, topic: str) -> List[str]:
        """Get all chapters for a specific subject and topic"""
        chapters = set()
        for doc in self.documents:
            if (doc.metadata.get("subject") == subject and 
                doc.metadata.get("topic") == topic):
                chapters.add(doc.metadata.get("chapter", "general"))
        return sorted(list(chapters))
    
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
        return {
            "total_documents": len(self.documents),
            "total_subjects": len(self.get_all_subjects()),
            "subjects": self.get_all_subjects(),
            "has_vector_store": self.vector_store is not None
        }
