"""Graph nodes: orchestrator, dispatcher, agent + tool nodes, synthesizer."""

from functools import lru_cache

from langchain.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.types import Send

from travel_planner import config
from travel_planner.agents.travel_scout import get_planner_tools
from travel_planner.llm import get_llm
from travel_planner.prompts import (
    FLIGHT_HOTEL_SEARCH_INSTRUCTIONS,
    NO_BOOKING_RULES,
    ROUTER_INSTRUCTIONS,
    ROUTER_MEMORY_RULES,
    SYNTHESIZER_PROMPT,
    TRAVEL_SCOUT_INSTRUCTIONS,
)
from travel_planner.graph.state import TravelPlannerState
from travel_planner.schemas import ClassificationResult
from travel_planner.tools import SEARCH_TOOLS


# ---------------------------------------------------------------------------
# LLMs augmented with tools
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _search_model_with_tools():
    return get_llm().bind_tools(SEARCH_TOOLS)


@lru_cache(maxsize=1)
def _planner_model_with_tools():
    return get_llm().bind_tools(get_planner_tools())


SEARCH_TOOLS_BY_NAME = {tool.name: tool for tool in SEARCH_TOOLS}

# Told to the agent once it has used its research rounds
ROUND_LIMIT_NOTE = (
    "Research limit reached: do not call any more tools. Answer now with the information "
    "you already have, and briefly note anything you could not verify."
)


def _tool_rounds(msgs: list) -> int:
    """How many times the agent has already asked for tools in this turn."""
    return sum(1 for m in msgs if getattr(m, "tool_calls", None))


def _answer_only(model_tools: list):
    """Same tools bound, but tool_choice="none": the model must answer in text."""
    return get_llm().bind_tools(model_tools, tool_choice="none")

# How many earlier messages (questions + answers) the router sees
HISTORY_WINDOW = 6


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def classify_query_parallel(state: TravelPlannerState) -> dict:
    """Router that can dispatch to MULTIPLE agents.

    Sees recent conversation history so it can turn follow-ups like
    "Major historical sites there" into self-contained agent queries.
    """
    structured_llm = get_llm().with_structured_output(ClassificationResult)
    history = state.get("messages", [])[-HISTORY_WINDOW:]
    question = HumanMessage(content=state["user_query"])

    result = structured_llm.invoke(
        [SystemMessage(content=ROUTER_INSTRUCTIONS + ROUTER_MEMORY_RULES)]
        + history
        + [question]
    )

    return {
        "tasks": [task.model_dump() for task in result.tasks],
        "requires_synthesis": result.requires_synthesis,
        "messages": [question],  # remember the question for later turns
    }


def dispatch_to_agents(state: TravelPlannerState):
    """Use the Send API to dispatch tasks to agents in PARALLEL."""
    sends = []
    for task in state.get("tasks", []):
        worker_state = {
            "messages": state.get("messages", []),
            "user_query": task["user_query"],
            "tasks": state.get("tasks", []),
            "requires_synthesis": state.get("requires_synthesis", False),
            "agent_results": [],  # Each worker starts fresh
            "final_answer": "",
            "search_messages": [],  # Initialize empty for workers
            "planner_messages": [],  # Initialize empty for workers
        }

        if task["source"] == "search_agent_node":
            sends.append(Send("search_agent_node", worker_state))
        elif task["source"] == "planner_agent_node":
            sends.append(Send("planner_agent_node", worker_state))

    return sends


# ---------------------------------------------------------------------------
# Synthesizer
# ---------------------------------------------------------------------------

def synthesizer_node(state: TravelPlannerState) -> dict:
    """Combine results from parallel agents into a unified response."""
    agent_results = state.get("agent_results", [])

    if not agent_results:
        return {
            "final_answer": "I couldn't process your request.",
            "messages": [AIMessage(content="I couldn't process your request.")],
        }

    # If only one result, no synthesis needed
    if len(agent_results) == 1:
        result = agent_results[0]["result"]
        return {"final_answer": result, "messages": [AIMessage(content=result)]}

    # Format results for synthesis
    results_formatted = "\n\n" + "=" * 50 + "\n\n".join([
        f"**{r['agent'].upper()}**\nFocus: {r.get('focus', 'N/A')}\n\n{r['result']}"
        for r in agent_results
    ])

    response = get_llm().invoke([
        HumanMessage(content=SYNTHESIZER_PROMPT.format(
            query=state["user_query"],
            results=results_formatted,
        ))
    ])

    return {
        "final_answer": response.content,
        "messages": [AIMessage(content=response.content)],
    }


# ---------------------------------------------------------------------------
# Search agent (flights & hotels)
# ---------------------------------------------------------------------------

def search_agent_node(state: TravelPlannerState):
    """Search agent with its own message history."""
    search_msgs = state.get("search_messages", [])

    # First call (from the orchestrator): start the scratchpad with the query.
    # It is saved to state so the agent still sees it after tool calls.
    new_msgs = [] if search_msgs else [HumanMessage(content=state.get("user_query", ""))]

    messages = [SystemMessage(content=FLIGHT_HOTEL_SEARCH_INSTRUCTIONS + NO_BOOKING_RULES)] + search_msgs + new_msgs
    if _tool_rounds(search_msgs) < config.SEARCH_MAX_TOOL_ROUNDS:
        response = _search_model_with_tools().invoke(messages)
    else:
        response = _answer_only(SEARCH_TOOLS).invoke(messages + [SystemMessage(content=ROUND_LIMIT_NOTE)])

    return {
        "search_messages": new_msgs + [response],
        "agent_results": [{
            "agent": "search_agent",
            "focus": "flights and hotels",
            "result": response.content,
        }] if not response.tool_calls else [],
    }


def search_tool_node(state: TravelPlannerState):
    """Execute search tools."""
    search_msgs = state.get("search_messages", [])
    if not search_msgs:
        return {"search_messages": []}

    last_message = search_msgs[-1]
    if not getattr(last_message, "tool_calls", None):
        return {"search_messages": []}

    result = []
    for tool_call in last_message.tool_calls:
        tool = SEARCH_TOOLS_BY_NAME.get(tool_call["name"])
        if tool:
            observation = tool.invoke(tool_call["args"])
            result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))

    return {"search_messages": result}


# ---------------------------------------------------------------------------
# Planner agent (Travel Scout + itinerary sub-agent)
# ---------------------------------------------------------------------------

def planner_agent_node(state: TravelPlannerState):
    """Planner agent with its own message history."""
    planner_msgs = state.get("planner_messages", [])

    # First call (from the orchestrator): start the scratchpad with the query.
    # It is saved to state so the agent still sees it after tool calls.
    new_msgs = [] if planner_msgs else [HumanMessage(content=state.get("user_query", ""))]

    messages = [SystemMessage(content=TRAVEL_SCOUT_INSTRUCTIONS)] + planner_msgs + new_msgs
    if _tool_rounds(planner_msgs) < config.PLANNER_MAX_TOOL_ROUNDS:
        response = _planner_model_with_tools().invoke(messages)
    else:
        response = _answer_only(get_planner_tools()).invoke(messages + [SystemMessage(content=ROUND_LIMIT_NOTE)])

    return {
        "planner_messages": new_msgs + [response],
        "agent_results": [{
            "agent": "planner_agent",
            "focus": "travel planning and itinerary",
            "result": response.content,
        }] if not response.tool_calls else [],
    }


def planner_tool_node(state: TravelPlannerState):
    """Execute planner tools."""
    planner_msgs = state.get("planner_messages", [])
    if not planner_msgs:
        return {"planner_messages": []}

    last_message = planner_msgs[-1]
    if not getattr(last_message, "tool_calls", None):
        return {"planner_messages": []}

    planner_tools_by_name = {tool.name: tool for tool in get_planner_tools()}
    result = []
    for tool_call in last_message.tool_calls:
        tool = planner_tools_by_name.get(tool_call["name"])
        if tool:
            observation = tool.invoke(tool_call["args"])
            result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))

    return {"planner_messages": result}


# ---------------------------------------------------------------------------
# Conditional edges
# ---------------------------------------------------------------------------

def should_continue_for_search_agent(state: TravelPlannerState):
    search_msgs = state.get("search_messages", [])
    if search_msgs and getattr(search_msgs[-1], "tool_calls", None):
        return "search_tool_node"
    return "synthesizer"


def should_continue_for_planner_agent(state: TravelPlannerState):
    """Check if planner needs to continue tool loop."""
    planner_msgs = state.get("planner_messages", [])
    if planner_msgs and getattr(planner_msgs[-1], "tool_calls", None):
        return "planner_tool_node"
    return "synthesizer"
