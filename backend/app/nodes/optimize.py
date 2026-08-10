from ..core.models import optimizer_llm
from ..state import LinkedInState

from langchain_core.prompts import ChatPromptTemplate


def optimize_node(state: LinkedInState) -> dict:

    effective_feedback = state.get('human_feedback_override') or state.get('critique_feedback', "")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Rewrite the LinkedIn draft to directly address the critique feedback while preserving clean line-spacing and a punchy hook."),
        ("human", "Topic: {topic}\nCurrent Draft:\n{post_draft}\nCritique Feedback:\n{feedback}"),
    ])

    chain = prompt | optimizer_llm
    response = chain.invoke({
        "topic": state.get("topic", ""),
        "post_draft": state.get("post_draft", ""),
        "feedback": effective_feedback,
    })
    
    content = response.content
    if isinstance(content, list):
        content = content[0].get('text', '')

    return {
        "post_draft": content,
        "human_feedback_override": None
    }
