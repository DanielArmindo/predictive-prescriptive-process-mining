# Main
from .startup import lifespan
from .RWLock import RWLock

# Schemas
from .schemas.StatusModel import StatusModel
from .schemas.PredictRequest import PredictRequest
from .schemas.PredictContentRequest import PredictContentRequest
from .schemas.BackupRequest import BackupRequest
from .schemas.RetrainRequest import RetrainRequest
from .schemas.NewDatasetRequest import NewDatasetRequest

predict_models = []
rwlock = RWLock()
