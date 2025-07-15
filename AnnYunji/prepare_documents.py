# 2. LangChain에서 사용할 수 있도록 page_content + metadata를 묶어 Document 형태로 저장
# .jsonl 형태로 저장

import json
from api import fetch_openapi_data
from document_generator import build_page_content, build_metadata


def create_documents(num_pages=5, rows_per_page=20):
    all_documents = []

    for page in range(1, num_pages + 1):
        response = fetch_openapi_data(page=page, rows=rows_per_page)
        items = response.get("body", {}).get("items", [])

        for item_wrapper in items:
            item = item_wrapper.get("item", {})
            if not item:
                continue

            content = build_page_content(item)
            metadata = build_metadata(item)

            if content.strip():
                doc = {
                    "page_content": content,
                    "metadata": metadata
                }
                all_documents.append(doc)

    return all_documents

def save_documents_to_jsonl(documents, path="data/documents1.jsonl"):
    with open(path, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

# total_count=41717, rows_per_page=100,
if __name__ == "__main__":
    documents = create_documents(num_pages=10, rows_per_page=20)  # 총 200건 수집 예시
    save_documents_to_jsonl(documents)
    print(f"{len(documents)}건의 문서를 저장했습니다.")