import requests
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv('API_KEY')

def fetch_openapi_data(page=1, rows=10):
    url = 'http://apis.data.go.kr/1471000/HtfsInfoService03/getHtfsItem01'

    params = {
        "pageNo": str(page),
        "numOfRows": str(rows),
        "servicekey": API_KEY,
        "type": "json",
    }

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    return response.json()
