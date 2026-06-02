from fastapi import UploadFile, HTTPException
import io
import csv
from core.constants import MetadataDataset


async def validate_csv_file(file: UploadFile) -> str:
    if file.content_type != "text/csv":
        raise HTTPException(status_code=400, detail="O ficheiro deve ser CSV")

    content = await file.read()
    text = content.decode("utf-8")

    csv_file = io.StringIO(text)
    try:
        reader = csv.reader(csv_file, delimiter=';')
        header = next(reader)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao ler CSV: {str(e)}")

    if len(header) < 2:
        raise HTTPException(status_code=400, detail="O CSV deve ter colunas separadas por ';'")

    return text


# Checks whether the columns in the retraining dataset are valid
async def validate_csv_retrain(columns: MetadataDataset, file: UploadFile) -> None:
    content = await file.read()
    text = content.decode("utf-8")

    csv_file = io.StringIO(text)

    try:
        reader = csv.reader(csv_file, delimiter=';')
        header = next(reader)
        header = [col.strip() for col in header]

        required_columns = list(columns.datasetColumns.values())

        if columns.contentColumn:
            required_columns.append(columns.contentColumn)

        if columns.orderColumn:
            required_columns.append(columns.orderColumn)

        if columns.metadataColumns:
            required_columns.append(columns.metadataColumns.columnPhase)
            required_columns.append(columns.metadataColumns.columnReceptionTimestamp)
            required_columns.append(columns.metadataColumns.humanResources)
            required_columns.append(columns.metadataColumns.columnYear)

        # Check for missing columns
        missing_columns = [col for col in required_columns if col not in header]

        if missing_columns:
            raise HTTPException(
                status_code=400,
                detail=f"Colunas em falta no CSV: {missing_columns}"
            )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao ler CSV: {str(e)}")


def resolve_forks(arr):
    result = []
    i = 0
    n = len(arr)

    while i < n:
        val = arr[i]
        int_part = int(val)

        if val == int_part:
            result.append(val)
            i += 1
            continue

        # Find a block of consecutive decimal places
        j = i
        bloco = []
        while j < n and arr[j] != int(arr[j]):
            bloco.append(arr[j])
            j += 1

        # Group by whole number (4.x, 5.x, etc.)
        grupos = {}
        for v in bloco:
            key = int(v)
            grupos.setdefault(key, []).append(v)

        # Choose a group (e.g., the first one)
        escolhido = sorted(grupos.keys())[0]
        result.extend(grupos[escolhido])

        i = j

    return result
