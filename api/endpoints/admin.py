import json
from fastapi import APIRouter, Form, Header, HTTPException, File, UploadFile
from typing import Optional
import os
import dill
from typing import Optional
from datetime import datetime
from fastapi.responses import JSONResponse
from api import StatusModel, BackupRequest, rwlock, RetrainRequest, NewDatasetRequest
from .utils import validate_csv_file, validate_csv_retrain
from core import MetadataDataset, createDatasetModel, TypeModels
from core.backups import list_backups, check_exists_file, store_model
from core.builder import updateModel

router = APIRouter(
    prefix="/admin",
    tags=["admin"]
)


# List existing backups
@router.get("/backups")
def listBackups():
    itens = list_backups()
    if not itens:
        return StatusModel(success=False, message="Sem backups existentes !")
    return itens


# Create a new or restore a backup
@router.post("/backups")
async def create_backup(body: BackupRequest, restore_backup: Optional[bool] = Header(None)):
    # Create backup
    if not restore_backup:
        fileName = body.filename if body.filename else datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S")
        filePath = f"backups/{fileName}.pkl"

        from api import predict_models
        with open(filePath, "wb") as f:
            dill.dump(predict_models, f)

        responseMsg = f"Backup realizado com o nome {fileName}"
        return StatusModel(success=True, message=responseMsg)

    if body.filename == None or body.filename.strip() == "":
        return HTTPException(400, "Filename to restore models is empty !")

    # To restore a backup containing everything
    if not check_exists_file(body.filename):
        response = StatusModel(
            success=False, message="Ficheiro de backup nã́o existente !").model_dump()
        return JSONResponse(content=response, status_code=404)

    await rwlock.acquire_write()
    try:
        import api
        filePath = f"backups/{id}.pkl"
        with open(filePath, "rb") as f:
            api.predict_models = dill.load(f)

        return StatusModel(success=True, message="Modelos foram carregados com sucesso !")
    finally:
        rwlock.release_write()


# List the existing backups for a specific dataset
@router.get("/backups/{id}")
def list_individual_backups(id: str):
    itens = list_backups(id)
    if not itens:
        return StatusModel(success=False, message=f"Sem backups existentes para o dataset {id} !")
    return itens


# Save or load individual dataset
@router.post("/backups/{id}")
async def create_individual_backup(id: str, body: BackupRequest, restore_backup: Optional[bool] = Header(None)):
    import api

    if restore_backup:
        if body.filename == None or body.filename.strip() == "":
            return HTTPException(400, "Filename to restore model is empty !")

        if not check_exists_file(body.filename, id):
            response = StatusModel(
                success=False, message="Ficheiro de backup nã́o existente !").model_dump()
            return JSONResponse(content=response, status_code=404)

        await rwlock.acquire_write()
        try:
            model = next((item for item in api.predict_models if item["name"] == id), None)
            if model == None:
                return HTTPException(500, "Fail to restore model !")
            filePath = f"backups/{id}/{body.filename}.pkl"
            with open(filePath, "rb") as f:
                model["model"] = dill.load(f)
        finally:
            rwlock.release_write()

        return StatusModel(success=True, message="Modelo foi carregado com sucesso !")


    model = next((item for item in api.predict_models if item["name"] == id), None)
    if model == None:
        return HTTPException(500, "Fail to backup model !")

    fileName = body.filename if body.filename and body.filename.strip() != "" else datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S")

    dirPath = f"backups/{id}"

    os.makedirs(dirPath, exist_ok=True)

    filePath = f"{dirPath}/{fileName}.pkl"

    with open(filePath, "wb") as f:
        dill.dump(model["model"], f)

    responseMsg = f"Backup realizado com o nome {fileName}"
    return StatusModel(success=True, message=responseMsg)


# Retrain a model/dataset
@router.post("/retrain")
async def retrain(
    body: RetrainRequest,
    content_dataset: UploadFile = File(...), 
    manual_templates: Optional[UploadFile] = File(None)
):
    updateURL = await validate_csv_file(content_dataset)
    id = body.model.replace("_content", "") if "_content" in body.model else body.model
    id_content_model = id if "_content" in id else f"{id}_content"

    import api

    model_api = next((item for item in api.predict_models if item["name"] == id), None)
    if model_api == None:
        return HTTPException(500, "Fail to backup model !")

    model_api_content = next((item for item in api.predict_models if item["name"] == id_content_model), None)

    # Check whether the retraining dataset contains the columns from the original dataset
    await validate_csv_retrain(model_api.metadata, content_dataset)

    templates = None

    if manual_templates is not None:
        templates = await validate_csv_file(manual_templates)

    try:
        model, content_model = await updateModel(model_api['metadata'], updateURL, templates, model_api.metadata.metadataColumns)
        if model == None:
            return HTTPException(400, "Error retraining predict activity model !")
    except:
        return HTTPException(400, "Error retraining models !")


    await rwlock.acquire_write()
    try:
        # Replace the template in the variable
        model_api["model"] = model
        if content_model and model_api_content != None:
            model_api_content["model"] = content_model
        if content_model and model_api_content == None:
            api.predict_models.append({
                "name": id_content_model,
                "model": content_model,
                "metadata": model_api["metadata"]
            })

        # Save to file
        fileName = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filePath = f"backups/{id}/{fileName}.pkl"

        with open(filePath, "wb") as f:
            dill.dump(model_api, f)

        if content_model:
            store_data = model_api_content if model_api_content != None else {
                "name": id_content_model,
                "model": content_model,
                "metadata": model_api["metadata"]
            }
            fileContentPath = f"backups/{id_content_model}/{fileName}.pkl"
            with open(fileContentPath, "wb") as f:
                dill.dump(store_data, f)

    finally:
        rwlock.release_write()

    msg = f"Modelo {id} foi ajustado com com sucesso !" if not content_model else f"Os modelos {
        id} e {id_content_model} foram ajustados com com sucesso !"
    return StatusModel(success=True, message=msg)


@router.post("/new_dataset")
async def new_dataset(
    new_item: str = Form(...), 
    content_dataset: UploadFile = File(...), 
    manual_templates: Optional[UploadFile] = File(None)
):
    body = NewDatasetRequest(**json.loads(new_item))
    content = await validate_csv_file(content_dataset)
    templates = None

    if manual_templates is not None:
        templates = await validate_csv_file(manual_templates)

    dataset_data = MetadataDataset(
        body.name, 
        body.type.value, 
        {
            "case_id": body.case_id,
            "activity_key": body.activity_key,
            "timestamp_key": body.timestamp_key,
            "start_timestamp_key": body.start_timestamp_key
        }, 
        body.contentColumn,
        body.orderColumn
    )

    if body.type != TypeModels.NORMAL and body.type != TypeModels.EDOC:
        return HTTPException(400, "Error type of modal must be between 'normal' or 'edoc' type !")

    try:
        model, model_content = await createDatasetModel(dataset_data, content, templates, body.metadataColumns)
        if model == None:
            return HTTPException(400, "Error training predict activity model !")
    except:
        return HTTPException(400, "Error training models !")

    await rwlock.acquire_write()
    import api
    try:
        model_act= {
            "name": dataset_data.name,
            "model": model,
            "metadata": dataset_data
        }

        api.predict_models.append(model_act)
        store_model(model_act, dataset_data.name)

        if model_content != None:
            model_content = {
                "name": f"{dataset_data.name}_content",
                "model": model_content,
                "metadata": dataset_data
            }
            api.predict_models.append(model_content)
            store_model(model_content, f"{dataset_data.name}_content")
    finally:
        rwlock.release_write()

    msg = f"Modelo(s) {dataset_data.name} criado(s) !"
    return StatusModel(success=True, message=msg)
