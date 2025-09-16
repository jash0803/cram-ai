import streamlit as st
import os
import json
from datetime import datetime, timedelta
from streamlit_option_menu import option_menu
from streamlit_chat import message

# Import our custom modules
from utils.pdf_processor import PDFProcessor
from utils.vector_store import VectorStoreManager
from utils.ai_helpers import AIHelpers
from config import *

# Resolve OpenAI API key: Streamlit secrets → env var → config.py
resolved_api_key = None
try:
    # st.secrets may raise if not configured (e.g., local without secrets.toml)
    resolved_api_key = st.secrets.get("OPENAI_API_KEY", None)  # type: ignore[attr-defined]
except Exception:
    resolved_api_key = None

if not resolved_api_key:
    resolved_api_key = os.getenv("OPENAI_API_KEY") or OPENAI_API_KEY

if resolved_api_key:
    os.environ["OPENAI_API_KEY"] = resolved_api_key

# Page configuration
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = VectorStoreManager()
    st.session_state.vector_store.load_vector_store()

if 'pdf_processor' not in st.session_state:
    st.session_state.pdf_processor = PDFProcessor()

if 'ai_helpers' not in st.session_state:
    st.session_state.ai_helpers = AIHelpers()

# Check API key configuration
if not os.environ.get("OPENAI_API_KEY"):
    st.error("⚠️ OpenAI API key not configured!")
    st.info("Set `OPENAI_API_KEY` in `.streamlit/secrets.toml` or as an environment variable.")
    st.code('[secrets]\nOPENAI_API_KEY="sk-..."')
    st.stop()

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'quiz_results' not in st.session_state:
    st.session_state.quiz_results = []

if 'flashcard_progress' not in st.session_state:
    st.session_state.flashcard_progress = {}

# Main navigation
with st.sidebar:
    selected = option_menu(
        menu_title="CramAI",
        options=["📁 Upload & Organize", "📚 Study Tools", "❓ Quiz Mode", "💬 Ask Questions", "🃏 Flashcards", "📊 Progress"],
        icons=["upload", "book", "question-circle", "chat", "card-text", "graph-up"],
        menu_icon="brain",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": {"color": "orange", "font-size": "25px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "#02ab21"},
        }
    )

# Upload & Organize Page
if selected == "📁 Upload & Organize":
    st.title("📁 Upload & Organize Your Study Materials")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Upload PDF Documents")
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload your study materials in PDF format"
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                st.write(f"📄 {uploaded_file.name}")
                
                # Organization inputs
                col_subject, col_topic, col_chapter = st.columns(3)
                
                with col_subject:
                    subject = st.selectbox(
                        f"Subject for {uploaded_file.name}",
                        DEFAULT_SUBJECTS,
                        key=f"subject_{uploaded_file.name}"
                    )
                
                with col_topic:
                    topic = st.text_input(
                        f"Topic for {uploaded_file.name}",
                        value="General",
                        key=f"topic_{uploaded_file.name}"
                    )
                
                with col_chapter:
                    chapter = st.text_input(
                        f"Chapter for {uploaded_file.name}",
                        value="Chapter 1",
                        key=f"chapter_{uploaded_file.name}"
                    )
                
                # Process button
                if st.button(f"Process {uploaded_file.name}", key=f"process_{uploaded_file.name}"):
                    with st.spinner("Processing PDF..."):
                        # Process PDF
                        processed_data = st.session_state.pdf_processor.process_pdf(
                            uploaded_file, uploaded_file.name
                        )
                        
                        # Add to vector store
                        metadata = {
                            "filename": uploaded_file.name,
                            "subject": subject,
                            "topic": topic,
                            "chapter": chapter,
                            "upload_date": datetime.now().isoformat()
                        }
                        
                        st.session_state.vector_store.add_documents(
                            processed_data["chunks"], metadata
                        )
                        
                        # Save vector store
                        st.session_state.vector_store.save_vector_store()
                        
                        st.success(f"✅ Successfully processed {uploaded_file.name}")
                        st.info(f"📊 Extracted {processed_data['chunk_count']} chunks, {processed_data['word_count']} words")
    
    with col2:
        st.subheader("📊 Your Library")
        stats = st.session_state.vector_store.get_stats()
        
        st.metric("Total Documents", stats["total_documents"])
        st.metric("Subjects", stats["total_subjects"])
        
        if stats["subjects"]:
            st.write("**Subjects:**")
            for subject in stats["subjects"]:
                st.write(f"• {subject}")

# Study Tools Page
elif selected == "📚 Study Tools":
    st.title("📚 AI Study Tools")
    
    # Get available subjects
    subjects = st.session_state.vector_store.get_all_subjects()
    
    if not subjects:
        st.warning("No documents uploaded yet. Please upload some PDFs first.")
    else:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            selected_subject = st.selectbox("Select Subject", subjects)
            
            # Get topics for selected subject
            topics = st.session_state.vector_store.get_topics_by_subject(selected_subject)
            selected_topic = st.selectbox("Select Topic", topics) if topics else None
            
            # Get chapters for selected topic
            if selected_topic:
                chapters = st.session_state.vector_store.get_chapters_by_topic(selected_subject, selected_topic)
                selected_chapter = st.selectbox("Select Chapter", chapters) if chapters else None
            else:
                selected_chapter = None
        
        with col2:
            if selected_subject and selected_topic and selected_chapter:
                # Filter documents
                filter_dict = {
                    "subject": selected_subject,
                    "topic": selected_topic,
                    "chapter": selected_chapter
                }
                
                # Get relevant documents
                relevant_docs = st.session_state.vector_store.search_similar(
                    f"{selected_subject} {selected_topic} {selected_chapter}",
                    k=10,
                    filter_dict=filter_dict
                )
                
                if relevant_docs:
                    st.subheader("📖 Content Summary")
                    
                    # Summary generation
                    summary_type = st.selectbox("Summary Type", SUMMARY_TYPES)
                    
                    if st.button("Generate Summary"):
                        with st.spinner("Generating summary..."):
                            summary = st.session_state.ai_helpers.generate_summary(
                                relevant_docs, summary_type
                            )
                            st.markdown("### Generated Summary")
                            st.write(summary)
                    
                    # Question generation
                    st.subheader("❓ Generate Questions")
                    
                    col_qtype, col_diff, col_count = st.columns(3)
                    
                    with col_qtype:
                        question_type = st.selectbox("Question Type", QUIZ_TYPES)
                    
                    with col_diff:
                        difficulty = st.selectbox("Difficulty", QUIZ_DIFFICULTY_LEVELS)
                    
                    with col_count:
                        question_count = st.slider("Number of Questions", 1, 10, 5)
                    
                    if st.button("Generate Questions"):
                        with st.spinner("Generating questions..."):
                            questions = st.session_state.ai_helpers.generate_questions(
                                relevant_docs, question_type, difficulty, question_count
                            )
                            
                            if questions:
                                st.markdown("### Generated Questions")
                                for i, q in enumerate(questions, 1):
                                    st.write(f"**Q{i}:** {q.get('question', 'N/A')}")
                                    if 'options' in q:
                                        for opt, val in q['options'].items():
                                            st.write(f"  {opt}. {val}")
                                    if 'answer' in q:
                                        with st.expander(f"Answer {i}"):
                                            st.write(q['answer'])
                                    if 'explanation' in q:
                                        with st.expander(f"Explanation {i}"):
                                            st.write(q['explanation'])
                                    st.write("---")
                else:
                    st.warning("No content found for the selected filters.")

# Quiz Mode Page
elif selected == "❓ Quiz Mode":
    st.title("❓ Interactive Quiz Mode")
    
    subjects = st.session_state.vector_store.get_all_subjects()
    
    if not subjects:
        st.warning("No documents uploaded yet. Please upload some PDFs first.")
    else:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            quiz_subject = st.selectbox("Quiz Subject", subjects, key="quiz_subject")
            topics = st.session_state.vector_store.get_topics_by_subject(quiz_subject)
            quiz_topic = st.selectbox("Quiz Topic", topics, key="quiz_topic") if topics else None
            
            if quiz_topic:
                chapters = st.session_state.vector_store.get_chapters_by_topic(quiz_subject, quiz_topic)
                quiz_chapter = st.selectbox("Quiz Chapter", chapters, key="quiz_chapter") if chapters else None
            else:
                quiz_chapter = None
            
            quiz_difficulty = st.selectbox("Difficulty", QUIZ_DIFFICULTY_LEVELS, key="quiz_difficulty")
            quiz_type = st.selectbox("Question Type", QUIZ_TYPES, key="quiz_type")
            quiz_count = st.slider("Number of Questions", 1, 20, 5, key="quiz_count")
            
            timed_mode = st.checkbox("Enable Timed Mode", key="timed_mode")
            time_limit = None
            if timed_mode:
                time_limit = st.slider("Time Limit (minutes)", 1, 60, 10, key="time_limit")
        
        with col2:
            if st.button("Start Quiz", key="start_quiz"):
                if quiz_subject and quiz_topic and quiz_chapter:
                    # Generate quiz questions
                    filter_dict = {
                        "subject": quiz_subject,
                        "topic": quiz_topic,
                        "chapter": quiz_chapter
                    }
                    
                    relevant_docs = st.session_state.vector_store.search_similar(
                        f"{quiz_subject} {quiz_topic} {quiz_chapter}",
                        k=10,
                        filter_dict=filter_dict
                    )
                    
                    if relevant_docs:
                        with st.spinner("Generating quiz..."):
                            quiz_questions = st.session_state.ai_helpers.generate_questions(
                                relevant_docs, quiz_type, quiz_difficulty, quiz_count
                            )
                        
                        if quiz_questions:
                            st.session_state.current_quiz = {
                                "questions": quiz_questions,
                                "answers": {},
                                "start_time": datetime.now(),
                                "time_limit": time_limit,
                                "subject": quiz_subject,
                                "topic": quiz_topic,
                                "chapter": quiz_chapter
                            }
                            st.rerun()
                    else:
                        st.error("No content found for quiz generation.")
                else:
                    st.error("Please select subject, topic, and chapter.")
        
        # Display current quiz
        if 'current_quiz' in st.session_state:
            quiz = st.session_state.current_quiz
            
            st.subheader(f"Quiz: {quiz['subject']} - {quiz['topic']} - {quiz['chapter']}")
            
            # Timer display
            if quiz['time_limit']:
                elapsed = datetime.now() - quiz['start_time']
                remaining = timedelta(minutes=quiz['time_limit']) - elapsed
                if remaining.total_seconds() > 0:
                    st.info(f"⏰ Time Remaining: {remaining}")
                else:
                    st.error("⏰ Time's up!")
            
            # Quiz questions
            for i, question in enumerate(quiz['questions']):
                st.write(f"**Question {i+1}:** {question.get('question', 'N/A')}")
                
                if 'options' in question:
                    # Multiple choice
                    answer = st.radio(
                        f"Answer {i+1}",
                        options=list(question['options'].keys()),
                        format_func=lambda x: f"{x}. {question['options'][x]}",
                        key=f"quiz_answer_{i}"
                    )
                    quiz['answers'][i] = answer
                else:
                    # Open-ended
                    answer = st.text_area(f"Your Answer {i+1}", key=f"quiz_answer_{i}")
                    quiz['answers'][i] = answer
                
                st.write("---")
            
            col_submit, col_show = st.columns(2)
            
            with col_submit:
                if st.button("Submit Quiz"):
                    # Calculate score
                    correct = 0
                    total = len(quiz['questions'])
                    
                    for i, question in enumerate(quiz['questions']):
                        user_answer = quiz['answers'].get(i, "")
                        if 'options' in question:
                            if user_answer == question.get('correct_answer'):
                                correct += 1
                        # For open-ended questions, we'll mark as correct for now
                        # In a real app, you'd want more sophisticated scoring
                    
                    score = (correct / total) * 100
                    
                    # Save result
                    result = {
                        "timestamp": datetime.now().isoformat(),
                        "subject": quiz['subject'],
                        "topic": quiz['topic'],
                        "chapter": quiz['chapter'],
                        "score": score,
                        "total_questions": total,
                        "correct_answers": correct
                    }
                    
                    st.session_state.quiz_results.append(result)
                    
                    st.success(f"Quiz completed! Score: {score:.1f}% ({correct}/{total})")
                    
                    # Show answers
                    with st.expander("View Answers"):
                        for i, question in enumerate(quiz['questions']):
                            st.write(f"**Q{i+1}:** {question.get('question', 'N/A')}")
                            if 'options' in question:
                                st.write(f"**Correct Answer:** {question.get('correct_answer', 'N/A')}")
                            if 'explanation' in question:
                                st.write(f"**Explanation:** {question['explanation']}")
                            st.write("---")
                    
                    # Clear current quiz
                    del st.session_state.current_quiz
            
            with col_show:
                if st.button("Show Answers"):
                    with st.expander("Quiz Answers"):
                        for i, question in enumerate(quiz['questions']):
                            st.write(f"**Q{i+1}:** {question.get('question', 'N/A')}")
                            if 'options' in question:
                                st.write(f"**Correct Answer:** {question.get('correct_answer', 'N/A')}")
                            if 'explanation' in question:
                                st.write(f"**Explanation:** {question['explanation']}")
                            st.write("---")

# Ask Questions Page
elif selected == "💬 Ask Questions":
    st.title("💬 Ask Questions")
    
    # Chat interface
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    for i, (user_msg, bot_msg) in enumerate(st.session_state.chat_history):
        message(user_msg, is_user=True, key=f"user_{i}")
        message(bot_msg, key=f"bot_{i}")
    
    # Chat input
    user_input = st.text_input("Ask a question about your study materials:", key="chat_input")
    
    if st.button("Send") and user_input:
        # Search for relevant documents
        relevant_docs = st.session_state.vector_store.search_similar(user_input, k=5)
        
        if relevant_docs:
            # Generate answer
            with st.spinner("Thinking..."):
                answer = st.session_state.ai_helpers.answer_question(user_input, relevant_docs)
            
            # Add to chat history
            st.session_state.chat_history.append((user_input, answer))
            
            # Display new messages
            message(user_input, is_user=True, key=f"user_{len(st.session_state.chat_history)-1}")
            message(answer, key=f"bot_{len(st.session_state.chat_history)-1}")
            
            st.rerun()
        else:
            st.warning("No relevant content found. Please upload some study materials first.")

# Flashcards Page
elif selected == "🃏 Flashcards":
    st.title("🃏 Flashcards")
    
    subjects = st.session_state.vector_store.get_all_subjects()
    
    if not subjects:
        st.warning("No documents uploaded yet. Please upload some PDFs first.")
    else:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            flashcard_subject = st.selectbox("Subject", subjects, key="flashcard_subject")
            topics = st.session_state.vector_store.get_topics_by_subject(flashcard_subject)
            flashcard_topic = st.selectbox("Topic", topics, key="flashcard_topic") if topics else None
            
            if flashcard_topic:
                chapters = st.session_state.vector_store.get_chapters_by_topic(flashcard_subject, flashcard_topic)
                flashcard_chapter = st.selectbox("Chapter", chapters, key="flashcard_chapter") if chapters else None
            else:
                flashcard_chapter = None
            
            flashcard_count = st.slider("Number of Flashcards", 5, 20, 10, key="flashcard_count")
            
            if st.button("Generate Flashcards", key="generate_flashcards"):
                if flashcard_subject and flashcard_topic and flashcard_chapter:
                    # Get relevant documents
                    filter_dict = {
                        "subject": flashcard_subject,
                        "topic": flashcard_topic,
                        "chapter": flashcard_chapter
                    }
                    
                    relevant_docs = st.session_state.vector_store.search_similar(
                        f"{flashcard_subject} {flashcard_topic} {flashcard_chapter}",
                        k=10,
                        filter_dict=filter_dict
                    )
                    
                    if relevant_docs:
                        with st.spinner("Generating flashcards..."):
                            flashcards = st.session_state.ai_helpers.generate_flashcards(
                                relevant_docs, flashcard_count
                            )
                        
                        if flashcards:
                            st.session_state.current_flashcards = flashcards
                            st.session_state.flashcard_index = 0
                            st.session_state.show_answer = False
                            st.rerun()
                    else:
                        st.error("No content found for flashcard generation.")
                else:
                    st.error("Please select subject, topic, and chapter.")
        
        with col2:
            if 'current_flashcards' in st.session_state:
                flashcards = st.session_state.current_flashcards
                current_index = st.session_state.flashcard_index
                show_answer = st.session_state.show_answer
                
                if current_index < len(flashcards):
                    current_card = flashcards[current_index]
                    
                    st.subheader(f"Flashcard {current_index + 1} of {len(flashcards)}")
                    
                    # Progress bar
                    progress = (current_index + 1) / len(flashcards)
                    st.progress(progress)
                    
                    # Card display
                    st.markdown("### Front")
                    st.info(current_card.get('front', 'No front content'))
                    
                    if show_answer:
                        st.markdown("### Back")
                        st.success(current_card.get('back', 'No back content'))
                        
                        col_next, col_prev = st.columns(2)
                        
                        with col_next:
                            if st.button("Next Card", key="next_card"):
                                st.session_state.flashcard_index += 1
                                st.session_state.show_answer = False
                                st.rerun()
                        
                        with col_prev:
                            if st.button("Previous Card", key="prev_card"):
                                if st.session_state.flashcard_index > 0:
                                    st.session_state.flashcard_index -= 1
                                    st.session_state.show_answer = False
                                    st.rerun()
                    else:
                        if st.button("Show Answer", key="show_answer"):
                            st.session_state.show_answer = True
                            st.rerun()
                else:
                    st.success("🎉 You've completed all flashcards!")
                    if st.button("Start Over", key="restart_flashcards"):
                        st.session_state.flashcard_index = 0
                        st.session_state.show_answer = False
                        st.rerun()

# Progress Page
elif selected == "📊 Progress":
    st.title("📊 Your Study Progress")
    
    # Quiz results
    if st.session_state.quiz_results:
        st.subheader("📈 Quiz Performance")
        
        # Convert to DataFrame for better visualization
        import pandas as pd
        
        df = pd.DataFrame(st.session_state.quiz_results)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Recent quizzes
        st.write("**Recent Quiz Results:**")
        recent_quizzes = df.tail(10)
        
        for _, quiz in recent_quizzes.iterrows():
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Subject", quiz['subject'])
            with col2:
                st.metric("Score", f"{quiz['score']:.1f}%")
            with col3:
                st.metric("Questions", f"{quiz['correct_answers']}/{quiz['total_questions']}")
            with col4:
                st.metric("Date", quiz['timestamp'].strftime("%m/%d"))
            st.write("---")
        
        # Performance by subject
        if len(df) > 1:
            st.subheader("📊 Performance by Subject")
            subject_scores = df.groupby('subject')['score'].mean().sort_values(ascending=False)
            
            for subject, avg_score in subject_scores.items():
                st.write(f"**{subject}:** {avg_score:.1f}% average")
    else:
        st.info("No quiz results yet. Take some quizzes to see your progress here!")
    
    # Study statistics
    st.subheader("📚 Study Statistics")
    stats = st.session_state.vector_store.get_stats()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Documents", stats['total_documents'])
    
    with col2:
        st.metric("Subjects Studied", stats['total_subjects'])
    
    with col3:
        st.metric("Quizzes Taken", len(st.session_state.quiz_results))
    
    # Subject breakdown
    if stats['subjects']:
        st.subheader("📖 Subjects in Your Library")
        for subject in stats['subjects']:
            topics = st.session_state.vector_store.get_topics_by_subject(subject)
            st.write(f"**{subject}:** {len(topics)} topics")

# Footer
st.markdown("---")
st.markdown("🧠 **CramAI** - Study Smarter, Not Harder | Built with Streamlit & LangChain")
