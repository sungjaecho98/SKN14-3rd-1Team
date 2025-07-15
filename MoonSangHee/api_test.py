import requests, json, math
from pprint import pprint

num_of_rows = 10

params = {
    "pageNo":    "1",          # 페이지 번호
    "numOfRows": str(num_of_rows),         # 한 페이지 결과 수
    "ServiceKey": decoding, # Swagger에 표시된 정확한 이름(ServiceKey)
    "type":      "json",       # 응답 포맷(xml/json) – default: xml
    # (필요시 API별 추가 파라미터를 여기에 더합니다)
}

response = requests.get(url, params=params, timeout=10)
response.raise_for_status()

body = response.json()["body"]
total_count = int(body.get("totalCount", 0))
total_pages = math.ceil(total_count / num_of_rows)
# for page in range(0, total_count, num_of_rows):
# for page in range(0, total_pages+1):
for page in range(0, 3):
    params["pageNo"] = str(page)
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    items = resp.json()["body"]["items"]["item"]
    # items 처리(저장 등)
    print(f"{page}페이지: {len(items)}개 처리")

with open("output.jsonl", "w", encoding="utf-8") as f:
    for page in range(1, total_pages+1):
        params["pageNo"] = str(page)
        resp = requests.get(url, params=params, timeout=10)
        items = resp.json().get("response", {}).get("body", {}).get("items", {}).get("item", [])
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
# pprint(response.json())
