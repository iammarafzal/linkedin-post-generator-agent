from ..state import LinkedInState

def human_review_node(state: LinkedInState) -> dict:
    # This node is a pass-through that acts as a breakpoint for human review.
    # The LangGraph workflow will pause here due to 'interrupt_before'.
    return {}
