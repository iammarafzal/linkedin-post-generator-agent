from typing import Literal, Optional
from pydantic import BaseModel, Field


class LinkedInPostSchema(BaseModel):
    post_draft: str = Field(
        description="The content of the linkedin post draft."
    )


class LinkedInEvaluation(BaseModel):
    evaluation: Literal["approved", "needs_improvement"] = Field(
            description="Set to 'approved' if the post has a compelling hook, high readability, strong value delivery, and a natural CTA. Otherwise 'needs_improvement'."
        )
    feedback: str = Field(
            description="Actionable critique detailing fixes (line spacing, hook punchiness, CTA)."
        )


class GenerateRequest(BaseModel):
    thread_id: str
    topic: str
    target_audience: Optional[str] = "Tech Professionals"
    tone: Optional[str] = "Educational & Authentic"


class ResumeRequest(BaseModel):
    thread_id: str
    action: str  # "approve", "override_feedback", or "direct_edit"
    custom_feedback: Optional[str] = None
    direct_draft_edit: Optional[str] = None
    