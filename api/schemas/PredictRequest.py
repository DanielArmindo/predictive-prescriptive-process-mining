from pydantic import BaseModel


class PredictRequest(BaseModel):
    model: str
    inProcess: bool = False
    prefix: list[str] = []
