import os
from datetime import datetime
import dill


def list_backups(directory: str | None = None):
    dirPath = 'backups' if not directory else f"backups/{directory}"
    if not os.path.isdir(dirPath):
        return []
    files = [
        f for f in os.listdir(dirPath)
        if os.path.isfile(os.path.join(dirPath, f))
    ]
    itens = []
    for file in files:
        itens.append(file.replace(".pkl", ""))

    return sorted(itens, key=str.lower)


def check_exists_file(filename: str, directory: str | None = None) -> bool:
    dirPath = 'backups' if not directory else f"backups/{directory}"
    files = os.listdir(dirPath)
    for file in files:
        formated_file = file.replace(".pkl", "")
        if formated_file == filename:
            return True

    return False


def store_model(content: dict, name: str) -> None:
    fileName = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dirPath = f"backups/{name}"
    os.makedirs(dirPath, exist_ok=True)
    filePath = f"{dirPath}/{fileName}.pkl"

    with open(filePath, "wb") as f:
        dill.dump(content, f)
