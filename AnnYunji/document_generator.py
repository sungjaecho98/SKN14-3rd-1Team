# 1. 데이터 정제 + 자연어 문서 생성
#- normalize_numbering — 숫자 표현 정규화
#- build_page_content — 자연어 설명 문단 생성
#- build_metadata — 메타데이터 추출

import re
from api import fetch_openapi_data

# 숫자 표현을 모두 '숫자)' 형태로 정규화 (①, ❶, ⑴, 1. 등)
def normalize_numbering(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'[\u2460-\u2473]', lambda m: f"{ord(m.group()) - 9311})", text)   # ① ~ ⑳
    text = re.sub(r'[\u2776-\u277F]', lambda m: f"{ord(m.group()) - 10101})", text)  # ❶ ~ ❿
    text = re.sub(r'[\u2488-\u249B]', lambda m: f"{ord(m.group()) - 9343})", text)   # ⑴ ~ ⒛
    text = re.sub(r'(\d+)[\s]*[.)]', lambda m: f"{int(m.group(1))})", text)          # 1), 1. 등
    return text


def build_page_content(item: dict) -> str:
    parts = []

    product = str(item.get("PRDUCT", "") or "").strip() # 제품명
    entrps = str(item.get("ENTRPS", "") or "").strip() # 업소명
    main_fnctn = str(item.get("MAIN_FNCTN", "") or "").strip() # 주요기능
    base_standard = str(item.get("BASE_STANDARD", "") or "").strip() # 기준규격
    prsrv_pd = str(item.get("PRSRV_PD", "") or "").strip() # 보존 및 유통기준
    distb_pd = str(item.get("DISTB_PD", "") or "").strip() # 소비 및 유통기한
    srv_use = str(item.get("SRV_USE", "") or "").strip() # 섭취량 & 섭취방법


    if product and entrps:
        parts.append(f"'{product}'은 {entrps}에서 제조한 건강기능식품 원료입니다.")
    elif product:
        parts.append(f"'{product}'은 건강기능식품 원료입니다.")

    if main_fnctn:
        parts.append(f"이 제품은 {normalize_numbering(main_fnctn)} 등의 기능이 있습니다.")

    if base_standard:
        parts.append(f"기준 규격은 다음과 같습니다: {normalize_numbering(base_standard)}")

    if prsrv_pd:
        parts.append(f"보존 및 유통기준은 {prsrv_pd}입니다.")

    if distb_pd:
        parts.append(f"소비 및 유통기한은 {distb_pd}입니다.")

    if srv_use:
        parts.append(f"섭취량 & 섭취방법은 '{srv_use}'입니다.")

    return " ".join(parts)


def build_metadata(item: dict) -> dict:
    keys_to_extract = [
        "PRDUCT", "ENTRPS", "DISTB_PD",
        # "REGIST_DT", "STTEMNT_NO",
        # "SRV_USE", "PRSRV_PD"
    ]
    meta = {key: item[key] for key in keys_to_extract if item.get(key) is not None}
    return meta


# ────────────────────────────────────────────────
if __name__ == "__main__":
    data = fetch_openapi_data(page=1, rows=1)
    items = data.get("body", {}).get("items", [])

    if items:
        item = items[0]["item"]
        page_content = build_page_content(item)
        metadata = build_metadata(item)

        print("Page Content:\n")
        print(page_content)
        print("\nMetadata:\n")
        print(metadata)
    else:
        print("데이터가 없습니다.")