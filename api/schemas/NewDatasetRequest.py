from pydantic import BaseModel
from typing import Optional
from core.constants import MetadataColumnsFlowModel, TypeModels


class NewDatasetRequest(BaseModel):
    name: str
    type: TypeModels
    case_id: str
    activity_key: str
    timestamp_key: str
    start_timestamp_key: str
    contentColumn: Optional[str] = None
    orderColumn: str
    metadataColumns: Optional[MetadataColumnsFlowModel] = None
