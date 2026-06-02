from contextlib import asynccontextmanager
from fastapi import FastAPI
import dill
from core.backups import list_backups
import signal

# Content intended for system initialization before serving FastAPI (lifespan)


@asynccontextmanager
async def lifespan(app: FastAPI):
    import api 
    # Load the models of prediction
    signal.signal(signal.SIGINT, close_program)
    models = load_previous_models()
    if models:
        api.predict_models = models

    yield
    # Clean up the all models and release the resources
    api.predict_models.clear()


# Function to avoid training models from scratch and load a backup
def load_previous_models():
    backups = list_backups()
    # Get the most recent backup
    last_one = backups[-1] if backups else None

    if not backups:
        # print("\nNão existem backups para carregar\n")
        return None

    filePath = f"backups/{last_one}.pkl"
    with open(filePath, "rb") as f:
        models = dill.load(f)

    return models


def close_program(sig, frame):
    print("\n\nFechando programa...\n")
    exit()
