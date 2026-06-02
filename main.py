from fastapi import FastAPI
from api import StatusModel, lifespan, rwlock
from api.endpoints import predict_router, admin_router
from dotenv import load_dotenv


load_dotenv()

app = FastAPI(lifespan=lifespan)


# Display types of models/datasets to predict
@app.get("/status")
async def status():
    await rwlock.acquire_read()
    try:
        from api import predict_models
        # print(predict_models)
        response = []
        for item in predict_models:
            name = item['name']
            response.append(
                {"name": name, "type": "content" if "_content" in name else "normal"})
        return response
    finally:
        await rwlock.release_read()


@app.get("/health", response_model=StatusModel)
def ready():
    return StatusModel(success=True, message="Ok")


app.include_router(predict_router)
app.include_router(admin_router)
