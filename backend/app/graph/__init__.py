from ..state import LinkedInState
from ..nodes import generate_node, evaluate_node, optimize_node

from typing import Literal
from langgraph.graph import END, START, StateGraph


def route_evaluation(state: LinkedInState) -> Literal["approved", "needs_improvement"]:
    evaluation = state.get("evaluation", "needs_improvement")
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iteration", 3)

    if evaluation == "approved" or iteration_count >= max_iterations:
        return "approved"
    return "needs_improvement"


def create_graph(checkpointer=None):

    builder = StateGraph(LinkedInState)

    builder.add_node("generate_node", generate_node)
    builder.add_node("evaluate_node", evaluate_node)
    builder.add_node("optimize_node", optimize_node)

    builder.add_edge(START, "generate_node")
    builder.add_edge("generate_node", "evaluate_node")

    builder.add_conditional_edges(
        "evaluate_node",
        route_evaluation,
        {
            "approved": END,
            "needs_improvement": "optimize_node",
        },
    )

    builder.add_edge("optimize_node", "evaluate_node")

    return builder.compile(
        checkpointer=checkpointer, interrupt_before=["optimize_node"]
    )
