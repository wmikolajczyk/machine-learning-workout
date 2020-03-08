import io
import os
import requests
from http import HTTPStatus
from zipfile import ZipFile


DATASET_DIR = "dataset"


def get_data(dataset_dir=DATASET_DIR):
    url = """https://storage.googleapis.com/kaggle-competitions-data/kaggle-v2/5407/868283/bundle/archive.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1583963097&Signature=V1VsrqZZfQ%2Bg0PxuK5VvKo5498kCAz2rlvO9f3fHWa6ib6N6ot%2BdGqasGLwP69wKrhsbgcTQKX13BueMsa6id8EClhBVn7mrdB4GOOLnm0iV1U5gVXDiUNYbvbAhu3dGQY3Yto2a4oNpyJ9EhYOqtGFvIWGDSyhrFm%2Bf3BtySE%2FQs7EwZgJp0S3aMHCoCDcMlwXDvBrQh7rNQo9Ms%2FQ8iOaTUUL1DKKbnfAuyWnTaMQchpoD5P78IRt%2B2uIAQPYfpQx3l4W4lEH5Oo3yZNsQIBfeSBpmsgM5erFl8TRQVrPrhW5pcTjVTNCm8NWSveju2QbgWf8GcfvM6VEmpxzu8g%3D%3D&response-content-disposition=attachment%3B+filename%3Dhouse-prices-advanced-regression-techniques.zip"""
    response = requests.get(url)
    assert response.status_code == HTTPStatus.OK
    os.makedirs(dataset_dir, exist_ok=True)
    # use in memory bytes stream
    f = io.BytesIO(response.content)
    with ZipFile(f, 'r') as zipObj:
    	zipObj.extractall(path=dataset_dir)
