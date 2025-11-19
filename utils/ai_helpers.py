from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
import streamlit as st
from config import OPENAI_API_KEY

class AIHelpers:
    """AI-powered content generation helpers"""
    
    def __init__(self):
        if OPENAI_API_KEY:
            self.llm = ChatOpenAI(
                openai_api_key=OPENAI_API_KEY,
                model_name="gpt-3.5-turbo",
                temperature=0.7
            )
        else:
            self.llm = None
    
    def generate_summary(self, documents: List[Document], summary_type: str = "Quick Recap") -> str:
        """Generate summary from documents"""
        
        if not self.llm:
            return "OpenAI API key not configured. Please set your API key in the .env file."
        
        # Combine document content
        content = "\n\n".join([doc.page_content for doc in documents])
        
        # Define prompts for different summary types
        prompts = {
            "Quick Recap": """
            Create a concise summary of the following content. Focus on key points and main concepts.
            Keep it brief and easy to understand for quick review.
            
            Content: {content}
            
            Summary:
            """,
            
            "Deep Dive": """
            Create a comprehensive summary of the following content. Include detailed explanations,
            important concepts, relationships between ideas, and key takeaways.
            
            Content: {content}
            
            Detailed Summary:
            """,
            
            "Flashcard Format": """
            Create a summary in flashcard format. For each key concept, provide:
            1. The concept/term
            2. A brief definition or explanation
            
            Format as:
            Q: [Question/Concept]
            A: [Answer/Explanation]
            
            Content: {content}
            
            Flashcard Summary:
            """
        }
        
        prompt_template = PromptTemplate(
            input_variables=["content"],
            template=prompts.get(summary_type, prompts["Quick Recap"])
        )
        
        try:
            # Use modern LangChain approach
            chain = prompt_template | self.llm
            result = chain.invoke({"content": content})
            return result.content if hasattr(result, 'content') else str(result)
        except Exception as e:
            st.error(f"Error generating summary: {str(e)}")
            return "Error generating summary. Please try again."
    
    def generate_questions(self, documents: List[Document], question_type: str, difficulty: str = "Medium", count: int = 5) -> List[Dict[str, Any]]:
        """Generate questions from documents"""
        
        if not self.llm:
            return [{"question": "OpenAI API key not configured. Please set your API key in the .env file.", "answer": "N/A"}]
        
        content = "\n\n".join([doc.page_content for doc in documents])
        
        prompts = {
            "MCQ": """
            Generate {count} multiple choice questions of {difficulty} difficulty based on the following content.
            For each question, provide:
            1. The question
            2. 4 options (A, B, C, D)
            3. The correct answer
            4. A brief explanation
            
            Format as JSON:
            {{
                "questions": [
                    {{
                        "question": "Question text",
                        "options": {{
                            "A": "Option A",
                            "B": "Option B", 
                            "C": "Option C",
                            "D": "Option D"
                        }},
                        "correct_answer": "A",
                        "explanation": "Explanation text"
                    }}
                ]
            }}
            
            Content: {content}
            """,
            
            "Objective": """
            Generate {count} objective questions of {difficulty} difficulty based on the following content.
            These should be short answer questions that test factual knowledge.
            
            Format as JSON:
            {{
                "questions": [
                    {{
                        "question": "Question text",
                        "answer": "Correct answer"
                    }}
                ]
            }}
            
            Content: {content}
            """,
            
            "Detailed Q&A": """
            Generate {count} detailed questions of {difficulty} difficulty based on the following content.
            These should require comprehensive answers that demonstrate understanding.
            
            Format as JSON:
            {{
                "questions": [
                    {{
                        "question": "Question text",
                        "answer": "Detailed answer with explanations"
                    }}
                ]
            }}
            
            Content: {content}
            """,
            
            "Numerical/Problem-solving": """
            Generate {count} numerical or problem-solving questions of {difficulty} difficulty based on the following content.
            Include step-by-step solutions.
            
            Format as JSON:
            {{
                "questions": [
                    {{
                        "question": "Problem statement",
                        "solution": "Step-by-step solution",
                        "final_answer": "Final numerical answer"
                    }}
                ]
            }}
            
            Content: {content}
            """
        }
        
        prompt_template = PromptTemplate(
            input_variables=["content", "count", "difficulty"],
            template=prompts.get(question_type, prompts["MCQ"])
        )
        
        try:
            # Use modern LangChain approach
            chain = prompt_template | self.llm
            result = chain.invoke({"content": content, "count": count, "difficulty": difficulty})
            result_text = result.content if hasattr(result, 'content') else str(result)
            
            # Try to parse JSON response
            import json
            try:
                parsed_result = json.loads(result_text)
                return parsed_result.get("questions", [])
            except json.JSONDecodeError:
                # If JSON parsing fails, return raw result
                return [{"question": result_text, "answer": "See generated content above"}]
                
        except Exception as e:
            st.error(f"Error generating questions: {str(e)}")
            return []
    
    def generate_flashcards(self, documents: List[Document], count: int = 10) -> List[Dict[str, str]]:
        """Generate flashcards from documents"""
        
        if not self.llm:
            return [{"front": "OpenAI API key not configured", "back": "Please set your API key in the .env file"}]
        
        content = "\n\n".join([doc.page_content for doc in documents])
        
        prompt = """
        Generate {count} flashcards based on the following content.
        Each flashcard should have a clear question on the front and a concise answer on the back.
        
        Format as JSON:
        {{
            "flashcards": [
                {{
                    "front": "Question or term",
                    "back": "Answer or definition"
                }}
            ]
        }}
        
        Content: {content}
        """
        
        prompt_template = PromptTemplate(
            input_variables=["content", "count"],
            template=prompt
        )
        
        try:
            # Use modern LangChain approach
            chain = prompt_template | self.llm
            result = chain.invoke({"content": content, "count": count})
            result_text = result.content if hasattr(result, 'content') else str(result)
            
            import json
            try:
                parsed_result = json.loads(result_text)
                return parsed_result.get("flashcards", [])
            except json.JSONDecodeError:
                return []
                
        except Exception as e:
            st.error(f"Error generating flashcards: {str(e)}")
            return []
    
    def answer_question(self, question: str, documents: List[Document]) -> str:
        """Answer a question using RAG"""
        
        if not self.llm:
            return "OpenAI API key not configured. Please set your API key in the .env file."
        
        context = "\n\n".join([doc.page_content for doc in documents])
        
        prompt = """
        Answer the following question based on the provided context. 
        If the answer is not in the context, say so clearly.
        Provide a comprehensive and accurate answer.
        
        Context: {context}
        
        Question: {question}
        
        Answer:
        """
        
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=prompt
        )
        
        try:
            # Use modern LangChain approach
            chain = prompt_template | self.llm
            result = chain.invoke({"context": context, "question": question})
            return result.content if hasattr(result, 'content') else str(result)
        except Exception as e:
            st.error(f"Error answering question: {str(e)}")
            return "Error generating answer. Please try again."
