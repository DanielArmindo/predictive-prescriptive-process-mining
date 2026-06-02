from pydantic import BaseModel
from typing import Optional


class BackupRequest(BaseModel):
    filename: Optional[str] = None
