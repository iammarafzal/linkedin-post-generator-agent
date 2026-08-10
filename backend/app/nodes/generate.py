from ..core.models import structured_generator_llm
from ..state import LinkedInState

from langchain_core.prompts import ChatPromptTemplate

def generate_node(state: LinkedInState) -> dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert LinkedIn content strategist. Write a scroll-stopping post with 1-2 sentence paragraphs, high white space, strong value, and a CTA."),
        ("human", "Topic: {topic}\nTarget Audience: {target_audience}\nTone: {tone}")
    ])

    chain = prompt | structured_generator_llm
    response = chain.invoke(
        {
            "topic": state.get("topic", ""),
            "target_audience": state.get("target_audience", "Tech Professionals"),
            "tone": state.get("tone", "Educational & Authentic"),
        }
    )
    return {
        'post_draft': response.post_draft,
        'iteration_count': 0
    }
