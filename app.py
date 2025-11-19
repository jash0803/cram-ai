import streamlit as st
import os
import re
from datetime import datetime, timedelta
from uuid import uuid4

import pandas as pd
from streamlit_option_menu import option_menu
from streamlit_chat import message

# Import our custom modules
from utils.pdf_processor import PDFProcessor
from utils.vector_store import VectorStoreManager
from utils.ai_helpers import AIHelpers
from config import *


def sanitize_filename(*parts: str) -> str:
    """Create a filesystem-friendly filename from provided parts."""
    raw = "_".join([part for part in parts if part])
    safe = re.sub(r"[^A-Za-z0-9_\-]+", "_", raw)
    return safe.strip("_") or "export"


def format_questions_markdown(questions):
    """Convert generated questions into a markdown string for download."""
    lines = []
    for idx, question in enumerate(questions, 1):
        lines.append(f"### Question {idx}")
        lines.append(question.get("question", "N/A"))
        
        options = question.get("options")
        if isinstance(options, dict):
            lines.append("")
            for key, value in options.items():
                lines.append(f"- {key}. {value}")
        
        if question.get("answer"):
            lines.append("")
            lines.append(f"**Answer:** {question['answer']}")
        
        if question.get("correct_answer"):
            lines.append("")
            lines.append(f"**Correct Answer:** {question['correct_answer']}")
        
        if question.get("explanation"):
            lines.append("")
            lines.append(f"**Explanation:** {question['explanation']}")
        
        if question.get("solution"):
            lines.append("")
            lines.append("**Solution:**")
            lines.append(question["solution"])
        
        if question.get("final_answer"):
            lines.append("")
            lines.append(f"**Final Answer:** {question['final_answer']}")
        
        lines.append("\n---\n")
    
    return "\n".join(lines).strip()


def parse_tags(tag_string: str):
    """Convert comma separated tags into a clean list."""
    return [tag.strip() for tag in (tag_string or "").split(",") if tag.strip()]


def get_relevant_documents(query: str, document_ids=None, k: int = 10):
    """Helper to retrieve documents based on query and optional document filters."""
    if 'vector_store' not in st.session_state:
        return []
    search_query = query or "study materials"
    docs = st.session_state.vector_store.search_similar(
        search_query,
        k=k,
        document_ids=document_ids if document_ids else None
    )
    if not docs and document_ids:
        docs = st.session_state.vector_store.get_documents_by_ids(document_ids)
    return docs

# Page configuration (must be the first Streamlit call)
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

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

if 'study_plan' not in st.session_state:
    st.session_state.study_plan = []

if 'saved_summaries' not in st.session_state:
    st.session_state.saved_summaries = []

if 'saved_question_sets' not in st.session_state:
    st.session_state.saved_question_sets = []

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
                
                tag_input = st.text_input(
                    f"Tags for {uploaded_file.name} (comma separated, optional)",
                    value="",
                    key=f"tags_{uploaded_file.name}"
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
                            "display_name": uploaded_file.name,
                            "tags": parse_tags(tag_input),
                            "upload_date": datetime.now().isoformat(),
                            "document_id": str(uuid4()),
                            "word_count": processed_data["word_count"],
                            "chunk_count": processed_data["chunk_count"],
                            "source_type": "PDF"
                        }
                        
                        st.session_state.vector_store.add_documents(
                            processed_data["chunks"], metadata
                        )
                        
                        # Save vector store
                        st.session_state.vector_store.save_vector_store()
                        
                        st.success(f"✅ Successfully processed {uploaded_file.name}")
                        st.info(f"📊 Extracted {processed_data['chunk_count']} chunks, {processed_data['word_count']} words")
        
        st.markdown("---")
        st.subheader("✍️ Add Quick Notes")
        with st.form("manual_notes_form", clear_on_submit=True):
            note_title = st.text_input("Title", value="My Notes")
            note_tags = st.text_input("Tags (comma separated)", value="", key="notes_tags")
            note_content = st.text_area("Write or paste your notes", height=200)
            submitted_notes = st.form_submit_button("Save Notes to Library")
        
        if submitted_notes:
            if not note_content.strip():
                st.error("Please add some content before saving your notes.")
            else:
                with st.spinner("Adding notes to your library..."):
                    processed_notes = st.session_state.pdf_processor.process_text(
                        note_content,
                        source_name=note_title or "Manual Notes"
                    )
                    metadata = {
                        "filename": note_title or "Manual Notes",
                        "display_name": note_title or "Manual Notes",
                        "tags": parse_tags(note_tags),
                        "upload_date": datetime.now().isoformat(),
                        "document_id": str(uuid4()),
                        "word_count": processed_notes["word_count"],
                        "chunk_count": processed_notes["chunk_count"],
                        "source_type": "Notes"
                    }
                    st.session_state.vector_store.add_documents(
                        processed_notes["chunks"],
                        metadata
                    )
                    st.success("✅ Notes saved successfully!")
    
    with col2:
        st.subheader("📊 Your Library")
        stats = st.session_state.vector_store.get_stats()
        
        st.metric("Total Documents", stats.get("total_documents", 0))
        st.metric("Total Chunks", stats.get("total_chunks", 0))
        
        library_data = st.session_state.vector_store.get_document_library()
        
        st.markdown("### 📚 Library Details")
        if library_data:
            library_df = pd.DataFrame(library_data)
            search_query = st.text_input("Search by title or tags", "")
            
            filtered_df = library_df.copy().fillna("")
            if search_query:
                mask = filtered_df["Document"].str.contains(search_query, case=False, na=False) | \
                       filtered_df["Tags"].str.contains(search_query, case=False, na=False)
                filtered_df = filtered_df[mask]
            
            st.dataframe(filtered_df, use_container_width=True, hide_index=True)
            
            csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download library snapshot (.csv)",
                data=csv_bytes,
                file_name=f"{sanitize_filename('cram_ai_library', datetime.now().isoformat())}.csv",
                mime="text/csv"
            )
        else:
            st.info("No documents or notes added yet. Upload a PDF or add quick notes to get started.")

# Study Tools Page
elif selected == "📚 Study Tools":
    st.title("📚 AI Study Tools")
    
    library_data = st.session_state.vector_store.get_document_library()
    doc_lookup = {
        f"{entry['Document']} ({entry['Document ID']})": entry["Document ID"]
        for entry in library_data
        if entry.get("Document ID")
    }
    
    if not library_data:
        st.warning("No documents uploaded yet. Please upload some PDFs or add notes first.")
    else:
        focus_col, docs_col = st.columns([2, 1])
        
        with focus_col:
            study_focus = st.text_input(
                "Describe what you want to study",
                value="Key concepts from my materials"
            )
        
        with docs_col:
            selected_doc_labels = st.multiselect(
                "Limit to uploads (optional)",
                options=list(doc_lookup.keys())
            )
            selected_doc_ids = [doc_lookup[label] for label in selected_doc_labels]
        
        relevant_docs = get_relevant_documents(study_focus, document_ids=selected_doc_ids, k=10)
        
        if relevant_docs:
            focus_label = study_focus or "General Study Focus"
            st.subheader("📖 Content Summary")
            
            summary_type = st.selectbox("Summary Type", SUMMARY_TYPES)
            
            if st.button("Generate Summary"):
                with st.spinner("Generating summary..."):
                    summary = st.session_state.ai_helpers.generate_summary(
                        relevant_docs, summary_type
                    )
                    st.markdown("### Generated Summary")
                    st.write(summary)
                    
                    summary_record = {
                        "focus": focus_label,
                        "document_ids": selected_doc_ids,
                        "summary_type": summary_type,
                        "content": summary,
                        "generated_at": datetime.now().isoformat()
                    }
                    st.session_state.saved_summaries.append(summary_record)
                    filename = f"{sanitize_filename(focus_label, summary_type, 'summary')}.md"
                    st.download_button(
                        "Download summary (.md)",
                        data=summary.encode("utf-8"),
                        file_name=filename,
                        mime="text/markdown"
                    )
            
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
                        
                        st.session_state.saved_question_sets.append({
                            "focus": focus_label,
                            "document_ids": selected_doc_ids,
                            "question_type": question_type,
                            "difficulty": difficulty,
                            "questions": questions,
                            "generated_at": datetime.now().isoformat()
                        })
                        
                        questions_md = format_questions_markdown(questions)
                        q_filename = f"{sanitize_filename(focus_label, question_type, difficulty, 'questions')}.md"
                        st.download_button(
                            "Download questions (.md)",
                            data=questions_md.encode("utf-8"),
                            file_name=q_filename,
                            mime="text/markdown"
                        )
        else:
            st.warning("No relevant content found. Try adjusting your study description or upload more materials.")

# Quiz Mode Page
elif selected == "❓ Quiz Mode":
    st.title("❓ Interactive Quiz Mode")
    
    library_data = st.session_state.vector_store.get_document_library()
    doc_lookup = {
        f"{entry['Document']} ({entry['Document ID']})": entry["Document ID"]
        for entry in library_data
        if entry.get("Document ID")
    }
    
    if not library_data:
        st.warning("No documents uploaded yet. Please upload some PDFs or add notes first.")
    else:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            quiz_focus = st.text_input(
                "What should the quiz cover?",
                value="General review of my materials",
                key="quiz_focus"
            )
            
            quiz_doc_labels = st.multiselect(
                "Limit to uploads (optional)",
                options=list(doc_lookup.keys()),
                key="quiz_docs"
            )
            quiz_doc_ids = [doc_lookup[label] for label in quiz_doc_labels]
            
            quiz_difficulty = st.selectbox("Difficulty", QUIZ_DIFFICULTY_LEVELS, key="quiz_difficulty")
            quiz_type = st.selectbox("Question Type", QUIZ_TYPES, key="quiz_type")
            quiz_count = st.slider("Number of Questions", 1, 20, 5, key="quiz_count")
            
            timed_mode = st.checkbox("Enable Timed Mode", key="timed_mode")
            time_limit = None
            if timed_mode:
                time_limit = st.slider("Time Limit (minutes)", 1, 60, 10, key="time_limit")
        
        with col2:
            if st.button("Start Quiz", key="start_quiz"):
                if quiz_focus.strip():
                    relevant_docs = get_relevant_documents(quiz_focus, document_ids=quiz_doc_ids, k=10)
                    
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
                                "focus": quiz_focus.strip(),
                                "document_ids": quiz_doc_ids
                            }
                            st.rerun()
                        else:
                            st.error("Unable to generate quiz questions. Please try again.")
                    else:
                        st.error("No content found for quiz generation. Try a different description or upload more materials.")
                else:
                    st.error("Please describe what the quiz should cover.")
        
        # Display current quiz
        if 'current_quiz' in st.session_state:
            quiz = st.session_state.current_quiz
            
            st.subheader(f"Quiz Focus: {quiz['focus']}")
            
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
                    
                    score = (correct / total) * 100 if total else 0
                    
                    # Save result
                    result = {
                        "timestamp": datetime.now().isoformat(),
                        "focus": quiz['focus'],
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
    
    library_data = st.session_state.vector_store.get_document_library()
    doc_lookup = {
        f"{entry['Document']} ({entry['Document ID']})": entry["Document ID"]
        for entry in library_data
        if entry.get("Document ID")
    }
    
    if not library_data:
        st.warning("No documents uploaded yet. Please upload some PDFs or add notes first.")
    else:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            flashcard_focus = st.text_input(
                "What should the flashcards cover?",
                value="Key facts to remember",
                key="flashcard_focus"
            )
            
            flashcard_doc_labels = st.multiselect(
                "Limit to uploads (optional)",
                options=list(doc_lookup.keys()),
                key="flashcard_docs"
            )
            flashcard_doc_ids = [doc_lookup[label] for label in flashcard_doc_labels]
            
            flashcard_count = st.slider("Number of Flashcards", 5, 20, 10, key="flashcard_count")
            
            if st.button("Generate Flashcards", key="generate_flashcards"):
                if flashcard_focus.strip():
                    relevant_docs = get_relevant_documents(flashcard_focus, document_ids=flashcard_doc_ids, k=10)
                    
                    if relevant_docs:
                        with st.spinner("Generating flashcards..."):
                            flashcards = st.session_state.ai_helpers.generate_flashcards(
                                relevant_docs, flashcard_count
                            )
                        
                        if flashcards:
                            st.session_state.current_flashcards = flashcards
                            st.session_state.flashcard_index = 0
                            st.session_state.show_answer = False
                            st.session_state.current_flashcard_focus = flashcard_focus.strip()
                            st.rerun()
                        else:
                            st.error("Unable to generate flashcards. Please try again.")
                    else:
                        st.error("No content found for flashcard generation. Try a different description or upload more materials.")
                else:
                    st.error("Please describe what the flashcards should cover.")
        
        with col2:
            if 'current_flashcards' in st.session_state:
                flashcards = st.session_state.current_flashcards
                current_index = st.session_state.flashcard_index
                show_answer = st.session_state.show_answer
                
                if current_index < len(flashcards):
                    current_card = flashcards[current_index]
                    
                    st.subheader(f"Flashcard {current_index + 1} of {len(flashcards)}")
                    if st.session_state.get("current_flashcard_focus"):
                        st.caption(f"Focus: {st.session_state.current_flashcard_focus}")
                    
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
                        if st.button("Show Answer", key="show_answer_btn"):
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
        
        df = pd.DataFrame(st.session_state.quiz_results)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        if 'focus' not in df.columns:
            df['focus'] = "General Focus"
        
        # Recent quizzes
        st.write("**Recent Quiz Results:**")
        recent_quizzes = df.tail(10)
        
        for _, quiz in recent_quizzes.iterrows():
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Focus:** {quiz.get('focus', 'General Focus')}")
            with col2:
                st.metric("Score", f"{quiz['score']:.1f}%")
            with col3:
                st.metric("Questions", f"{quiz['correct_answers']}/{quiz['total_questions']}")
            st.caption(f"Completed on {quiz['timestamp'].strftime('%m/%d %H:%M')}")
            st.write("---")
        
        if len(df) > 1:
            st.subheader("📊 Average Score by Focus")
            focus_scores = df.groupby('focus')['score'].mean().sort_values(ascending=False)
            
            for focus, avg_score in focus_scores.items():
                st.write(f"**{focus}:** {avg_score:.1f}% average")
    else:
        st.info("No quiz results yet. Take some quizzes to see your progress here!")
    
    # Study statistics
    st.subheader("📚 Study Statistics")
    stats = st.session_state.vector_store.get_stats()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Documents", stats.get('total_documents', 0))
    
    with col2:
        st.metric("Total Chunks", stats.get('total_chunks', 0))
    
    with col3:
        st.metric("Quizzes Taken", len(st.session_state.quiz_results))

    st.subheader("🗓️ Study Planner")
    with st.form("study_plan_form", clear_on_submit=True):
        plan_title = st.text_input("Goal / Activity", value="Review key concepts")
        plan_focus = st.text_input("Focus Area / Notes", value="General focus", key="plan_focus_input")
        plan_due = st.date_input("Target Date", datetime.now().date())
        plan_goal_type = st.selectbox("Goal Type", ["Summary", "Quiz", "Flashcards", "Reading", "Notes"], key="plan_goal_type")
        plan_notes = st.text_area("Notes (optional)", key="plan_notes")
        submit_plan = st.form_submit_button("Add to Planner")
    
    if submit_plan:
        st.session_state.study_plan.append({
            "title": plan_title,
            "focus": plan_focus,
            "due": plan_due.isoformat(),
            "goal_type": plan_goal_type,
            "notes": plan_notes,
            "status": "Pending",
            "created_at": datetime.now().isoformat()
        })
        st.success("🗂️ Study goal added!")
    
    if st.session_state.study_plan:
        remove_indices = []
        for idx, plan in enumerate(st.session_state.study_plan):
            col_title, col_meta, col_status, col_actions = st.columns([3, 2, 1, 1])
            col_title.markdown(f"**{plan['title']}**  \n{plan.get('notes') or 'No extra notes'}")
            col_meta.write(f"Focus: {plan.get('focus', 'General focus')}")
            col_meta.write(f"Due: {plan['due']}")
            completed = col_status.checkbox(
                "Done",
                value=plan['status'] == "Done",
                key=f"plan_done_{idx}"
            )
            plan['status'] = "Done" if completed else "Pending"
            if col_actions.button("Remove", key=f"plan_remove_{idx}"):
                remove_indices.append(idx)
            col_actions.write(f"Status: {plan['status']}")
            st.write("---")
        
        for idx in sorted(remove_indices, reverse=True):
            st.session_state.study_plan.pop(idx)
    else:
        st.info("Create your first study goal to start planning!")
    
    st.subheader("📝 Saved Summaries & Question Sets")
    summaries_tab, questions_tab = st.tabs(["Summaries", "Question Sets"])
    
    with summaries_tab:
        if st.session_state.saved_summaries:
            for entry in reversed(st.session_state.saved_summaries[-5:]):
                label = f"{entry['summary_type']} – {entry.get('focus', 'General Focus')}"
                with st.expander(label):
                    st.write(entry['content'])
                    st.caption(f"Generated on {entry['generated_at']}")
        else:
            st.info("No summaries saved yet. Generate one from the Study Tools page.")
    
    with questions_tab:
        if st.session_state.saved_question_sets:
            for entry in reversed(st.session_state.saved_question_sets[-5:]):
                label = f"{entry['question_type']} ({entry['difficulty']}) – {entry.get('focus', 'General Focus')}"
                with st.expander(label):
                    for i, question in enumerate(entry['questions'], 1):
                        st.write(f"**Q{i}:** {question.get('question', 'N/A')}")
                        if 'options' in question:
                            for opt, text in question['options'].items():
                                st.write(f"- {opt}. {text}")
                        if 'answer' in question:
                            st.write(f"**Answer:** {question['answer']}")
                        if 'correct_answer' in question:
                            st.write(f"**Correct Answer:** {question['correct_answer']}")
                        if 'explanation' in question:
                            st.write(f"**Explanation:** {question['explanation']}")
                        st.write("")
                    st.caption(f"Generated on {entry['generated_at']}")
        else:
            st.info("No question sets saved yet. Generate some from Study Tools.")

# Footer
st.markdown("---")
st.markdown("🧠 **CramAI** - Study Smarter, Not Harder | Built with Streamlit & LangChain")
