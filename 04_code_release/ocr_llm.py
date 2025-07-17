from paddleocr import PaddleOCR
from PIL import Image
import numpy as np
from config import load_config
from openai import OpenAI
import os

class OCR_LLM():

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.openai_model_name = self.cfg['OPENAI_MODEL_NAME']

    
    def image_ocr(self, img_file):

        ocr = PaddleOCR(lang='korean')
        image = Image.open(img_file).convert("RGB")

        image_np = np.array(image)

        result_list = ocr.predict(image_np)

        if not result_list:
            print("OCR 결과가 없습니다.")
            exit()

        result_data = result_list[0]
        texts = result_data['rec_texts']

        return texts #문자열로 구성된 리스트로 반환
    
    
    def keyword_llm(self, word_list):
        client = OpenAI()
        text = ', '.join(word_list)

        prompt = f"""
        다음은 건강기능식품 관련 제품 정보에서 추출한 비정형 텍스트 리스트입니다:

        {text}

        이 정보를 기반으로 제품을 대표할 수 있는 의미 있는 키워드 5개를 뽑아주세요.
        가능하면 아래 기준을 고려하세요:
        - 들어온 순서는 중요성과는 무관합니다.
        - 제품명으로 생각되는 키워드 위주로 뽑아주세요.
        - 한국어로만 된 단어만 뽑아주세요
        - 영어로 된 단어는 제외해주세요.
        - 유아, 어린이, 임산부, 남성, 여성 등 특정 대상 나이 표현 단어 뽑아주세요.

        - '' 안에 있는 것은 하나의 단어로 취급하므로 절대 쪼개지 마세요.


        결과는 리스트 형태로 출력해주세요 (예: ['키워드1', '키워드2', ...]).
        제품명으로 생각되는 키워드를 앞쪽에 배치해주세요
        """

        response = client.chat.completions.create(model=self.openai_model_name, messages=[{"role": "user", "content": prompt}], temperature=0.3)

        return response.choices[0].message.content

    def ocr_to_llm(self, img_file):

        word_list = self.image_ocr(img_file)
        return self.keyword_llm(word_list)
