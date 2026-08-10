from ..core.models import structured_evaluator_llm
from ..state import LinkedInState

from langchain_core.prompts import ChatPromptTemplate

def evaluate_node(state: LinkedInState) -> dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert LinkedIn content strategist. Critically evaluate the draft against hook strength, line spacing readability, clear value, and engaging CTA."),
        ("human", "Topic: {topic}\nDraft:\n{post_draft}\nPrevious Feedback:\n{critique_feedback}")
    ])
    current_iteration = state.get('iteration_count', 0) + 1
    result = (prompt | structured_evaluator_llm).invoke({
        "topic": state.get('topic', ""),
        "post_draft": state.get("post_draft", ""),
        "critique_feedback": state.get("critique_feedback", "None"),
    })

    return {
        "evaluation": result.evaluation,
        "critique_feedback": result.feedback,
        "iteration_count": current_iteration,
    }
