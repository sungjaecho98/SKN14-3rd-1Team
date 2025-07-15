import requests, json, math, os
from pprint import pprint
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('API_KEY')

url = 'http://apis.data.go.kr/1471000/HtfsInfoService03/getHtfsItem01?'

num_of_rows = 10

params = {
    "pageNo":    "1",          # 페이지 번호
    "numOfRows": str(num_of_rows),         # 한 페이지 결과 수
    "ServiceKey": API_KEY, # Swagger에 표시된 정확한 이름(ServiceKey)
    "type":      "json",       # 응답 포맷(xml/json) – default: xml
    # (필요시 API별 추가 파라미터를 여기에 더합니다)
}

response = requests.get(url, params=params, timeout=10)
response.raise_for_status()

body = response.json()["body"]
total_count = int(body.get("totalCount", 0))
total_pages = math.ceil(total_count / num_of_rows)

items_section = body.get("items", [])

# for page in range(1, total_pages):
for page in range(1, 4):

    params["pageNo"] = str(page)
    # print(params['pageNo'])
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    body_ = resp.json()["body"]
    items_section = body_.get("items", [])
    
    if isinstance(items_section, list):
        items_list = [ entry["item"] for entry in items_section ]
    
    else:
        items_list = [ items_section["item"] ]

    # pprint(items_list)
    
    with open('output.jsonl', 'a+', encoding='utf-8') as f:
        for item in items_list:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"{page}페이지: {len(items_list)}개 처리")