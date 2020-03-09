import io
import os
from http import HTTPStatus
from zipfile import ZipFile

import requests

DATASET_DIR = "dataset"


def get_data(url, dataset_dir=DATASET_DIR):
    response = requests.get(url)
    assert response.status_code == HTTPStatus.OK
    os.makedirs(dataset_dir, exist_ok=True)
    # use in memory bytes stream
    f = io.BytesIO(response.content)
    with ZipFile(f, 'r') as zipObj:
        zipObj.extractall(path=dataset_dir)
