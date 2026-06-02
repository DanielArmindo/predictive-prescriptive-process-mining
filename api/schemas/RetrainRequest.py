from pydantic import BaseModel


class RetrainRequest(BaseModel):
    model: str
