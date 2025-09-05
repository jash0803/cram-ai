# 🧠 CramAI – Study Smarter, Not Harder

A comprehensive web application that uses **RAG (Retrieval-Augmented Generation)** to help students upload PDFs, organize by subject/topic, and instantly generate summaries, quizzes, and intelligent Q&A content for fast, effective last-minute studying.

## 🚀 Features

### ✅ Upload & Organize
- Upload PDFs or notes
- Tag them by **subject**, **topic**, and **chapter**
- Automatically extracted text and structured using NLP

### 📚 AI Study Tools
1. **Summary Generator**
   - Auto-generates short summaries per section
   - Choose between "Quick Recap," "Deep Dive," or "Flashcard Format"

2. **Question Generator**
   - AI-generated questions:
     - **MCQs**
     - **Objective**
     - **Detailed Q&A**
     - **Numerical/Problem-solving**
   - Categorized by Bloom's taxonomy (recall, analysis, application)

3. **Quiz Mode**
   - Select topic → choose difficulty level → start quiz
   - "Show Answer" button only reveals answers on click (promotes active recall)
   - Option to enable **timed mode** for mock exams

4. **Ask a Question (Chat Interface)**
   - Ask questions directly (like "Explain Newton's Laws" or "Give me 5 MCQs on Thermodynamics")
   - AI retrieves relevant sections + provides tailored response

5. **Flashcards**
   - Auto-generate flashcards from key points and definitions
   - Supports spaced repetition algorithm

6. **Progress Tracking**
   - Visual dashboard for completed topics, quiz scores, and weak areas

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd cram-ai
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   - Create a `.env` file in the root directory
   - Add your OpenAI API key:
     ```
     OPENAI_API_KEY=your_openai_api_key_here
     ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

## 📋 Requirements

- Python 3.8+
- OpenAI API key
- Streamlit
- LangChain
- PyPDF2
- FAISS
- Other dependencies listed in `requirements.txt`

## 🎯 Usage

1. **Upload Documents**: Go to "Upload & Organize" and upload your PDF study materials
2. **Organize Content**: Tag each document with subject, topic, and chapter
3. **Generate Summaries**: Use "Study Tools" to create summaries in different formats
4. **Create Quizzes**: Generate practice questions and take interactive quizzes
5. **Ask Questions**: Use the chat interface to ask specific questions about your materials
6. **Study with Flashcards**: Generate and study with AI-created flashcards
7. **Track Progress**: Monitor your quiz performance and study statistics

## 🏗️ Architecture

- **Frontend**: Streamlit web interface
- **Backend**: Python with LangChain for AI processing
- **Vector Store**: FAISS for document embeddings and similarity search
- **AI Models**: OpenAI GPT models for content generation
- **PDF Processing**: PyPDF2 for text extraction

## 📁 Project Structure

```
cram-ai/
├── app.py                 # Main Streamlit application
├── config.py             # Configuration settings
├── requirements.txt      # Python dependencies
├── utils/
│   ├── __init__.py
│   ├── pdf_processor.py  # PDF text extraction
│   ├── vector_store.py   # Vector store management
│   └── ai_helpers.py     # AI content generation
└── README.md
```

## 🔧 Configuration

Edit `config.py` to customize:
- Default subjects and topics
- Chunk sizes for text processing
- Quiz difficulty levels
- Summary types

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

If you encounter any issues or have questions, please open an issue on GitHub.

---

**Built with ❤️ using Streamlit and LangChain**
