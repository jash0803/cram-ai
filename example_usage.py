#!/usr/bin/env python3
"""
Example usage of CramAI components
"""

import os
from utils.pdf_processor import PDFProcessor
from utils.vector_store import VectorStoreManager
from utils.ai_helpers import AIHelpers
from config import OPENAI_API_KEY

def example_usage():
    """Demonstrate how to use CramAI components"""
    
    print("🧠 CramAI Example Usage")
    print("=" * 50)
    
    # Check if OpenAI API key is set
    if not OPENAI_API_KEY or OPENAI_API_KEY == "your_openai_api_key_here":
        print("❌ Please set your OpenAI API key in the .env file")
        return
    
    # Initialize components
    print("🔧 Initializing components...")
    pdf_processor = PDFProcessor()
    vector_store = VectorStoreManager()
    ai_helpers = AIHelpers()
    
    # Example: Process a PDF (you would need an actual PDF file)
    print("\n📄 PDF Processing Example:")
    print("To process a PDF, use:")
    print("processed_data = pdf_processor.process_pdf(pdf_file, 'filename.pdf')")
    
    # Example: Add documents to vector store
    print("\n🗄️ Vector Store Example:")
    print("To add documents to vector store:")
    print("""
    metadata = {
        'filename': 'example.pdf',
        'subject': 'Mathematics',
        'topic': 'Calculus',
        'chapter': 'Chapter 1'
    }
    vector_store.add_documents(text_chunks, metadata)
    """)
    
    # Example: Search for similar documents
    print("\n🔍 Search Example:")
    print("To search for similar documents:")
    print("results = vector_store.search_similar('calculus derivatives', k=5)")
    
    # Example: Generate summary
    print("\n📝 Summary Generation Example:")
    print("To generate a summary:")
    print("summary = ai_helpers.generate_summary(documents, 'Quick Recap')")
    
    # Example: Generate questions
    print("\n❓ Question Generation Example:")
    print("To generate questions:")
    print("questions = ai_helpers.generate_questions(documents, 'MCQ', 'Medium', 5)")
    
    # Example: Answer questions
    print("\n💬 Q&A Example:")
    print("To answer questions:")
    print("answer = ai_helpers.answer_question('What is calculus?', documents)")
    
    print("\n✅ Example usage completed!")
    print("Run 'streamlit run app.py' to start the web application")

if __name__ == "__main__":
    example_usage()
