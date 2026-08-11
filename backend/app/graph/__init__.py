from ..state import LinkedInState
from ..nodes import generate_node, evaluate_node, optimize_node, human_review_node

from typing import Literal
from langgraph.graph import END, START, StateGraph


def route_after_evaluation(state: LinkedInState) -> str:
    evaluation = state.get("evaluation", "needs_improvement")
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iteration", 3)

    if evaluation == "approved" or iteration_count >= max_iterations:
        return "human_review_node"
    return "optimize_node"


def route_after_human(state: LinkedInState) -> str:
    human_approved = state.get("human_approved", False)
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iteration", 3)

    if human_approved:
        return END
    elif not human_approved and iteration_count < max_iterations:
        return "optimize_node"
    return END


def create_graph(checkpointer=None):
    builder = StateGraph(LinkedInState)

    builder.add_node("generate_node", generate_node)
    builder.add_node("evaluate_node", evaluate_node)
    builder.add_node("optimize_node", optimize_node)
    builder.add_node("human_review_node", human_review_node)

    builder.add_edge(START, "generate_node")
    builder.add_edge("generate_node", "evaluate_node")
    builder.add_edge("optimize_node", "evaluate_node")

    builder.add_conditional_edges(
        "evaluate_node",
        route_after_evaluation,
        {
            "human_review_node": "human_review_node",
            "optimize_node": "optimize_node",
        },
    )

    builder.add_conditional_edges(
        "human_review_node",
        route_after_human,
        {
            "optimize_node": "optimize_node",
            END: END,
        },
    )

    return builder.compile(
        checkpointer=checkpointer, interrupt_before=["human_review_node"]
    )
