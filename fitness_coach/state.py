from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import NotRequired, TypedDict


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    route: NotRequired[str]
    confidence: NotRequired[float]
    response: NotRequired[str]
    needs_input: NotRequired[bool]  # True when an agent needs more info before it can act
