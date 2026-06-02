from pydantic import BaseModel


class PredictContentRequest(BaseModel):
    model: str
    activity: list[str]
    prefix_content: list[str] = []
