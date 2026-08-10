from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ..core.schemas import GenerateRequest, ResumeRequest
from ..graph import agent

app = FastAPI(title="LinkedIn Post Generator Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/generate")
async def start_generation(req: GenerateRequest):
    config = {"configurable": {"thread_id": req.thread_id}}
    initial_state = {
        "topic": req.topic,
        "target_audience": req.target_audience,
        "tone": req.tone,
        "max_iteration": 3,
    }

    result = agent.invoke(initial_state, config=config)
    state = agent.get_state(config)

    return {
        "status": "paused_for_human" if state.next else "completed",
        "next_step": state.next[0] if state.next else None,
        "current_state": state.values,
    }

@app.post("/api/resume")
async def resume_generation(req: ResumeRequest):
    config = {"configurable": {"thread_id": req.thread_id}}
    current_state = agent.get_state(config)

    if not current_state.next:
        raise HTTPException(
            status_code=400, detail="Thread is not in a paused state."
        )

    if req.action == "approve":
        agent.update_state(config, {"evaluation": "approved"})
        result = agent.invoke(None, config=config)
    elif req.action == "override_feedback":
        # Ingest custom user critique and trigger optimize node
        agent.update_state(
            config, {"human_feedback_override": req.custom_feedback}
        )
        result = agent.invoke(None, config=config)
    elif req.action == "direct_edit":
        # User manually edited the post draft
        agent.update_state(
            config,
            {"post_draft": req.direct_draft_edit, "evaluation": "approved"},
        )
        result = agent.invoke(None, config=config)

    updated_state = agent.get_state(config)

    return {
        "status": "paused_for_human" if updated_state.next else "completed",
        "next_step": updated_state.next[0] if updated_state.next else None,
        "current_state": updated_state.values,
    }
