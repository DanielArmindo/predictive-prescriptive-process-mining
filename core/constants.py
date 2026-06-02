from dataclasses import dataclass
from enum import Enum
from typing import Optional


class TypeModels(Enum):
    NORMAL = "normal"
    EDOC = "edoc"


@dataclass
class MetadataDataset():
    name: str
    type: str
    datasetColumns: dict[str, str]
    contentColumn: Optional[str]
    orderColumn: str
    metadataColumns: Optional[MetadataColumnsFlowModel] = None


@dataclass
class MetadataColumnsFlowModel():
    columnReceptionTimestamp: str
    columnYear: str
    humanResources: str
    columnPhase: str
