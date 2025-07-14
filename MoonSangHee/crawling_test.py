# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.chrome.options import Options
# import time

# # 💻 PC 사이즈 브라우저 창 설정
# options = Options()
# options.add_argument("window-size=1400,1000")  # 넓은 화면

# driver = webdriver.Chrome(options=options)

# try:
#     driver.get("https://data.mfds.go.kr/hid/main/main.do")
#     time.sleep(3)  # 로딩 기다림

#     # "기능성 정보" 탭 클릭 (PC 기준 상단 메뉴)
#     ftn_info_link = driver.find_element(By.XPATH, '//a[text()="기능성 정보"]')
#     ftn_info_link.click()
#     time.sleep(3)  # 페이지 전환 대기

#     # 현재 페이지 정보 출력
#     print("✅ 페이지 제목:", driver.title)
#     print("✅ 현재 URL:", driver.current_url)

# except Exception as e:
#     print("❌ 에러 발생:", e)

# input("엔터를 누르면 종료합니다...")
# driver.quit()

# <a href="/hid/opdaa01/ftnltInfoLst.do">기능성 정보</a>

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time, json
from bs4 import BeautifulSoup
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def save_page_data(page_data, filename="products.jsonl"):
    with open(filename, "a", encoding="utf-8") as f:
        for item in page_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def restore_page(driver, page):
    print(f"🔄 [페이지 복원 시도]: {page}")
    try:
        if page == 1:
            return

        # 블록 이동 횟수 계산
        block_count = (page - 1) // 10
        current = 10

        # 🔹 블록 반복 이동 (10단위로 이동)
        for b in range(block_count):
            # 현재 base_page (10, 20...) 클릭
            base_page = current
            btn = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, f'span[onclick="linkPage({base_page}); return false;"]'))
            )
            btn.click()
            time.sleep(1)

            # 🔹 삼각형 클릭
            next_block = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, 'span.page_next'))
            )
            next_block.click()
            time.sleep(1)

            current += 10

        # 🔹 마지막으로 원하는 페이지 클릭
        if page % 10 != 0:
            btn = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, f'span[onclick="linkPage({page}); return false;"]'))
            )
            btn.click()
            time.sleep(1.5)

        print(f"✅ 복원 페이지 이동 완료: {page}")

    except Exception as e:
        print(f"❌ 복원 실패 (page {page}):", e)

options = Options()
options.add_argument("window-size=1400,1000")
driver = webdriver.Chrome(options=options)

driver.get("https://data.mfds.go.kr/hid/opdaa01/ftnltInfoLst.do")
time.sleep(3)

try:
    # 인지기능/기억력 항목 클릭
    card = driver.find_element(By.XPATH, '//*[contains(text(), "인지기능/기억력")]')
    card.click()
    time.sleep(3)

    print("✅ 현재 페이지 제목:", driver.title)
    print("✅ 현재 URL:", driver.current_url)

    # "관련제품 보기" 탭 클릭 (li[data-tab='relatedPrdt'])
    tab = driver.find_element(By.CSS_SELECTOR, 'li[data-tab="relatedPrdt"]')
    tab.click()
    time.sleep(3)
    
    all_data = []
    # ✅ 페이지 버튼 클릭 후에 현재 페이지의 제품 리스트만큼 반복
    for page in range(10, 31):  # 원하는 페이지 수만큼 반복
        print(f"\n📄 {page}페이지 시작")

        products = WebDriverWait(driver, 5).until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'td.cursor_p')))
        print(f'총 {len(products)}개 제품')
        for i in range(len(products)):  # 각 페이지의 최대 10개 제품
            try:
                # 제품 목록 새로 로드
                products = WebDriverWait(driver, 5).until(
                    EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'td.cursor_p'))
                )
                product = products[i]
                name = product.text.strip()
                print(f"👉 {i+1}/{len(products)} 제품 클릭: {name}")
                product.click()
                time.sleep(2)

                # 상세정보 파싱
                soup = BeautifulSoup(driver.page_source, 'html.parser')
                tables = soup.find_all('table')
                data = {"제품명": name}
                if len(tables) >= 2:
                    for tr in tables[1].find_all("tr"):
                        tds = tr.find_all("td")
                        if len(tds) == 2:
                            key = tds[0].text.strip()
                            val = tds[1].text.strip()
                            data[key] = val
                    all_data.append(data)
                    print(data)
                    print(f"✅ 수집 완료: {name}")
                else:
                    print("❌ 테이블 없음")

                # 뒤로가기
                driver.back()
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, 'li[data-tab="relatedPrdt"]'))
                )
                time.sleep(0.5)

                # 탭 다시 클릭
                tab = driver.find_element(By.CSS_SELECTOR, 'li[data-tab="relatedPrdt"]')
                tab.click()
                time.sleep(0.5)

                # 제품 목록 다시 로딩
                WebDriverWait(driver, 10).until(
                    EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'td.cursor_p'))
                )

                # 제품 상세보기 후 뒤로간 뒤 페이지 복원
                if page != 1:
                    restore_page(driver, page)

            except Exception as e:
                print(f"❌ {i+1}번 제품 수집 실패:", e)
                break
    with open(f'products_page_{page}.jsonl', 'w', encoding='utf-8') as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"💾 저장 완료: page {page}")

except Exception as e:
    print("❌ 에러 발생:", e)

input("엔터를 누르면 종료합니다...")
driver.quit()


