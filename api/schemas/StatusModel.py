from pydantic import BaseModel


class StatusModel(BaseModel):
    success: bool
    message: str
