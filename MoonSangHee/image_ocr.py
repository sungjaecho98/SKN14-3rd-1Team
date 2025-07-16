from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pprint import pprint

img_path = './image10.jpeg'
ocr = PaddleOCR(lang='korean')

image = Image.open(img_path).convert("RGB")  # 항상 RGBA로 변환
# background = Image.new("RGBA", image.size, (255, 255, 255, 255))  # 흰색 + 완전 불투명
# img_no_alpha = Image.alpha_composite(background, image).convert("RGB")  # 최종 RGB

image_np = np.array(image)

result_list = ocr.predict(image_np) # 변수명을 result_list로 변경하여 혼동 방지

# 실제 필요한 데이터는 result_list의 첫 번째 요소인 딕셔너리 안에 있습니다.
# 만약 predict 결과가 항상 하나의 딕셔너리를 포함하는 리스트라면 result_data = result_list[0]으로 접근합니다.
if not result_list:
    print("OCR 결과가 없습니다.")
    exit()

result_data = result_list[0] # 첫 번째 (그리고 유일한) 결과 딕셔너리

draw = ImageDraw.Draw(image)
font_path = './NanumGothicBold.ttf' # 폰트 경로 확인
try:
    font = ImageFont.truetype(font_path, 24)
except IOError:
    print(f"오류: 폰트 파일 '{font_path}'을(를) 찾을 수 없거나 로드할 수 없습니다.")
    print("시스템에 폰트가 설치되어 있는지 확인하거나, 정확한 폰트 경로를 지정해주세요.")
    # 대체 폰트 사용 또는 프로그램 종료
    font = ImageFont.load_default() # 기본 폰트를 로드하여 계속 진행

# ----------------- 이 부분이 수정되었습니다 -----------------
# result_data 딕셔너리에서 직접 데이터 추출
boxes = result_data['dt_polys']    # 감지된 박스의 좌표 (ndarray 리스트)
texts = result_data['rec_texts']    # 인식된 텍스트 문자열 (리스트)
scores = result_data['rec_scores']  # 인식된 텍스트의 신뢰도 점수 (리스트)
# --------------------------------------------------------

# dt_polys의 각 박스는 4개의 (x,y) 좌표를 가진 numpy 배열입니다.
# 예를 들어 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
# 이 중 첫 번째 (top_left)와 세 번째 (bottom_right)를 사용합니다.

# for box, text, score in zip(boxes, texts, scores):
#     # box는 이미 numpy 배열이므로, map(int, ...) 대신 직접 int로 캐스팅
#     top_left = (int(box[0][0]), int(box[0][1]))       # [x1, y1]
#     bottom_right = (int(box[2][0]), int(box[2][1]))   # [x3, y3]

#     print(f"Detected text: {text} (Probability: {score:.2f})")

#     # 사각형 그리기 (RGB 색상으로 지정)
#     draw.rectangle([top_left, bottom_right], outline=(0, 255, 0), width=2)
    
#     # 텍스트 그리기 (RGB 색상으로 지정, 일반적으로 top_left에서 시작)
#     draw.text(top_left, text, font=font, fill=(255, 0, 0))

# image.save('result.png')
# new_text = ""
# print(texts)
# for i in texts:
#     new_text += i

print("OCR 결과가 result.png로 저장되었습니다.")
print(texts)

# text_list = ['vhpro', '비오틴 효모 더 채움', 'vhpro', 'Setn Yeast tne Plus', '비오틴 효모 더 채움', 'Bofin Yeast the Plus', '/바요/', '/비오/원', 'SR', '', '200ng×1.5008 (300g)']
# print(set(text_list))