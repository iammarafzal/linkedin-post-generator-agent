from typing import Literal, Optional, TypedDict

class LinkedInState(TypedDict, total=False):
    topic: str
    target_audience: str
    tone: str

    post_draft: str
    critique_feedback: str
    human_feedback_override: Optional[str]
    evaluation: Literal["approved", "needs_improvement"]

    iteration_count: int
    max_iteration: int
    