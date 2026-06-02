import json
from fastapi import APIRouter, Header, Body
from fastapi.responses import JSONResponse
from typing import Optional
from api import PredictRequest, StatusModel, PredictContentRequest, rwlock
from rapidfuzz import fuzz
from collections import Counter
from api.endpoints.utils import resolve_forks


router = APIRouter(
    prefix="/predict",
    tags=["predict"]
)


@router.post("/")
async def predict(body: PredictRequest, additional_metrics: Optional[bool] = Header(None)):
    model = body.model
    prefix = body.prefix
    inProcess = body.inProcess
    await rwlock.acquire_read()
    try:
        from api import predict_models

        item = next((x for x in predict_models if x["name"] == model), None)
        if item != None:
            model_predict = item['model']
            if not additional_metrics:
                probs = model_predict.predict_next(
                    prefix, k=5, in_subprocess=inProcess)
                return probs
            else:
                probs = model_predict.predict_next_with_context(
                    prefix, k=5, in_subprocess=inProcess)
                return probs

        response = StatusModel(success=False, message="Modelo não existe !").dict()
        return JSONResponse(content=response, status_code=404)

    finally:
        await rwlock.release_read()


@router.post("/content")
async def predict_content(body: PredictContentRequest, only_templates: Optional[bool] = Header(None)):
    await rwlock.acquire_read()
    try:
        from api import predict_models

        model = f"{body.model}_content"
        activity = body.activity
        prefix = body.prefix_content

        item = next((x for x in predict_models if x["name"] == model), None)
        if item != None:
            model_predict = item['model']
            if not only_templates:
                probs = await model_predict.fill_template(
                    activity, prefix)
            else:
                probs = model_predict.predict_templates(activity, k=3)

            return probs

        response = StatusModel(success=False, message="Modelo não existe !").dict()
        return JSONResponse(content=response, status_code=404)

    finally:
        await rwlock.release_read()


@router.post("/main")
async def main_predict(body: str = Body(..., media_type="text/plain")):
    await rwlock.acquire_read()
    try:
        current_process: dict = json.loads(body)

        prefix = []
        prefix_content = []
        model: None | str = None

        best_name = None
        best_score = 0

        if not current_process or (current_process and 'Result' in current_process and len(current_process['Result']) == 0):
            return JSONResponse(content="O conteudo do pedido não presente para previsão !", status_code=404)

        arr_ordered = sorted(current_process['Result'], key=lambda x: float(x["Key"]['StageOrder']))
        arr_order = [float(x["Key"]['StageOrder']) for x in arr_ordered]
        formated_arr_order = resolve_forks(arr_order)

        code: str = arr_ordered[0].get('Key', {}).get('FlowKey', {}).get('Code', "")
        isEdoc = "edoc" in code.strip().lower()

        original_name: str | None = arr_ordered[0].get('Key', {}).get('FlowKey', {}).get("Code", None)

        if original_name is None:
            return JSONResponse(content="O conteudo do pedido não presente para previsão !", status_code=404)

        from api import predict_models
        
        for item in predict_models:
            if "_content" in item['name']:
                continue
            name = item['name']

            score = fuzz.partial_ratio(original_name.lower(), name.lower())

            if score > best_score:
                best_score = score
                best_name = name
            
        model = best_name if best_score >= 60 else None


        final_arr = []
        contador = Counter(formated_arr_order)

        for item in arr_ordered:
            val = float(item["Key"]["StageOrder"])
            if contador[val] > 0:
                final_arr.append(item)
                contador[val] -= 1

        for act in final_arr:
            prefix.append(act['Name']) if not isEdoc else prefix.append(act['Intervenient']['Name'])
            if 'Text' in act and act['Text'] != "":
                prefix_content.append(act['Text'])

        if model is None:
            return JSONResponse(content="O conteudo do pedido não presente para previsão !", status_code=404)

    except json.JSONDecodeError:
        return JSONResponse(content="O conteudo do processo não está no formato json", status_code=404)
    except Exception:
        return JSONResponse(content="Erro ao efetuar previsão de conteúdo", status_code=400)
    finally:
        await rwlock.release_read()

    bodyRequest = PredictContentRequest(
        model=model,
        activity=prefix,
        prefix_content=prefix_content
    )

    # print(bodyRequest)

    return await predict_content(bodyRequest, False)
