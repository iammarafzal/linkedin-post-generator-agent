import json
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool
from sse_starlette.sse import EventSourceResponse

from ..core.schemas import GenerateRequest, ResumeRequest
from ..graph import create_graph
from ..middleware import limiter, setup_middleware


load_dotenv()


ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

if ENVIRONMENT == "production":
    DATABASE_URL = os.getenv("DATABASE_URL")
else:
    # Use local postgres in development
    DATABASE_URL = os.getenv("LOCAL_DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/linkedin_agent")
agent = None
pool = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent, pool

    # Initialize connection pool
    pool = AsyncConnectionPool(
        conninfo=DATABASE_URL,
        max_size=10,
        kwargs={"autocommit": True, "prepare_threshold": None},
        open=False,
    )

    await pool.open()

    # Initialize AsyncPostgresSaver and create checkpoint tables if they don't exist
    checkpointer = AsyncPostgresSaver(pool)
    await checkpointer.setup()

    # Compile agent with persistent checkpointer
    agent = create_graph(checkpointer=checkpointer)

    yield

    # Clean up pool on shutdown
    if pool:
        await pool.close()


app = FastAPI(title="LinkedIn Post Generator Agent API", lifespan=lifespan)
setup_middleware(app)

@app.get("/health")
@limiter.limit("60/minute")
async def health_check(request: Request, response: Response):
    return {"status": "ok", "message": "Backend is awake and running!"}

@app.post("/api/generate")
@limiter.limit("5/minute")
async def start_generation(request: Request, response: Response, req: GenerateRequest):
    config = {"configurable": {"thread_id": req.thread_id}}
    initial_state = {
        "topic": req.topic,
        "target_audience": req.target_audience,
        "tone": req.tone,
        "max_iteration": 3,
    }

    result = await agent.ainvoke(initial_state, config=config)
    state = await agent.aget_state(config)

    return {
        "status": "paused_for_human" if state.next else "completed",
        "next_step": state.next[0] if state.next else None,
        "current_state": state.values,
    }

@app.post("/api/resume")
@limiter.limit("10/minute")
async def resume_generation(request: Request, response: Response, req: ResumeRequest):
    config = {"configurable": {"thread_id": req.thread_id}}
    current_state = await agent.aget_state(config)

    if not current_state.next:
        raise HTTPException(
            status_code=400, detail="Thread is not in a paused state."
        )

    if req.action == "approve":
        await agent.aupdate_state(config, {"human_approved": True})
        result = await agent.ainvoke(None, config=config)
    elif req.action == "request_changes":
        # Ingest custom user critique and trigger optimize node
        await agent.aupdate_state(
            config, {"human_approved": False, "human_feedback_override": req.custom_feedback}
        )
        result = await agent.ainvoke(None, config=config)
    elif req.action == "direct_edit":
        # User manually edited the post draft
        await agent.aupdate_state(
            config,
            {"post_draft": req.direct_draft_edit, "human_approved": True},
        )
        result = await agent.ainvoke(None, config=config)

    updated_state = await agent.aget_state(config)

    return {
        "status": "paused_for_human" if updated_state.next else "completed",
        "next_step": updated_state.next[0] if updated_state.next else None,
        "current_state": updated_state.values,
    }


@app.get("/api/stream/{thread_id}")
@limiter.limit("15/minute")
async def stream_generation(request: Request, response: Response, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}

    async def event_generator():
        state = await agent.aget_state(config)

        async for event in agent.astream_events(None, config, version="v2"):
            if await request.is_disconnected():
                break

            kind = event.get("event")
            node_name = event.get("name")

            if kind == "on_chain_start" and node_name in [
                "generate_node",
                "evaluate_node",
                "optimize_node"
            ]:
                yield {
                    "event": "node_start",
                    "data": json.dumps(
                        {"node": node_name, "status": f"Running {node_name}..."}
                    )
                }

            elif kind == "on_chain_end" and node_name in [
                "generate_node",
                "evaluate_node",
                "optimize_node"
            ]:
                output_data = event.get("data", {}).get("output", {})
                yield {
                    "event": "node_end",
                    "data": json.dumps(
                        {"node": node_name, "output": output_data}
                    )
                }

        final_state = await agent.aget_state(config)
        yield {
            "event": "complete",
            "data": json.dumps({
                "status": ("paused_for_human" if final_state.next else "completed"),
                "next_step": (final_state.next[0] if final_state.next else None),
                "current_state": final_state.values,
            })
        }

    return EventSourceResponse(event_generator())