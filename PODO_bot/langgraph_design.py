import operator
import os
from typing import Annotated, TypedDict, Sequence, Any, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# 1. State (Memory)
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    task_info: str
    image_size: tuple
    plan: Optional[str]
    human_approved: Optional[bool]

# 2. Nodes (Agents)
def main_agent(state: AgentState) -> dict:
    # Boss agent. Talk to user, check if plan is needed.
    pass

def planning_agent(state: AgentState) -> dict:
    # Choose models, check resources, create plan.
    pass

def human_review(state: AgentState) -> dict:
    # Human in the loop (HITL) node.
    pass

def preprocess_agent(state: AgentState) -> dict:
    # Use tools: resize, crop
    pass

def predict_agent(state: AgentState) -> dict:
    # Use models to predict
    pass

def postprocess_agent(state: AgentState) -> dict:
    # Clean up results
    pass

# 3. Image Predictor Sub-graph
image_builder = StateGraph(AgentState)
image_builder.add_node("preprocess_agent", preprocess_agent)
image_builder.add_node("predict_agent", predict_agent)
image_builder.add_node("postprocess_agent", postprocess_agent)

image_builder.set_entry_point("preprocess_agent")
image_builder.add_edge("preprocess_agent", "predict_agent")
image_builder.add_edge("predict_agent", "postprocess_agent")
image_builder.add_edge("postprocess_agent", END)

image_predictor_graph = image_builder.compile()

# 4. Main Graph
builder = StateGraph(AgentState)

# Add agents to map
builder.add_node("main_agent", main_agent)
builder.add_node("planning_agent", planning_agent)
builder.add_node("human_review", human_review)
builder.add_node("image_predictor", image_predictor_graph)

def main_routing(state: AgentState) -> str:
    plan = state.get("plan")
    
    # If no plan, route to planner
    if not plan:
        return "to_planner"
    
    # If plan exists, we need human approval
    return "to_human_review"

def review_routing(state: AgentState) -> str:
    approved = state.get("human_approved")
    if approved:
        return "to_predictor"
    else:
        return "to_main"

# Draw lines
builder.set_entry_point("main_agent")

builder.add_conditional_edges(
    "main_agent",
    main_routing,
    {
        "to_planner": "planning_agent",
        "to_human_review": "human_review"
    }
)

builder.add_edge("planning_agent", "main_agent")

builder.add_conditional_edges(
    "human_review",
    review_routing,
    {
        "to_predictor": "image_predictor",
        "to_main": "main_agent"
    }
)

builder.add_edge("image_predictor", END)

# Build it with Memory and Interrupt
memory = MemorySaver()
graph = builder.compile(interrupt_before=["human_review"], checkpointer=memory)

# 4. Draw PNG
def save_picture():
    try:
        # Need internet to ask mermaid to draw PNG
        img_data = graph.get_graph(xray=1).draw_mermaid_png()
        save_path = os.path.join(os.path.dirname(__file__), "langgraph_design.png")
        with open(save_path, "wb") as f:
            f.write(img_data)
        print(f"Ugga! Me saved map to {save_path}")
    except Exception as e:
        print(f"Me failed to draw. Cave too dark: {e}")

if __name__ == "__main__":
    save_picture()
