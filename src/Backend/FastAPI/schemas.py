from pydantic import BaseModel


class AgentRequest(BaseModel):
    message: str
    session_id: str | None = None