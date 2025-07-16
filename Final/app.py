import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')
import streamlit.components.v1 as components
# Python이 이미 한 번 임포트한 모듈을 캐시해두기 때문에 리로드 필요
import importlib
# import RAG_Chatbot as rag_chatbot
# importlib.reload(rag_chatbot)
from rag_chatbot import RAG_Chatbot
import recommand  as recommand
importlib.reload(recommand)
from recommand import get_recommendation_from_web
from config import load_config

cfg = load_config()
rag_chatbot = RAG_Chatbot(cfg)

# 페이지 설정
st.set_page_config(
    page_title="🌿 NutriWise - AI 영양제 추천",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 사용자 정의 CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    }
    
    .ingredient-tag {
        background: #e6fffa;
        color: #234e52;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin: 0.2rem;
        display: inline-block;
    }
    
    .warning-box {
        background: #fef5e7;
        border: 2px solid #f6ad55;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    
    .user-message {
        background: #e6f3ff;
        text-align: right;
    }
    
    .bot-message {
        background: #f0f9ff;
    }
</style>
""", unsafe_allow_html=True)

# 메인 헤더
st.markdown("""
<div class="main-header">
    <h1>🌿 NutriWise</h1>
    <p style="font-size: 1.2rem; margin-top: 0.5rem;">AI 기반 개인 맞춤형 영양제 추천 시스템</p>
</div>
""", unsafe_allow_html=True)

# # 영양 성분 분석 데이터
# @st.cache_data
# def get_nutrition_analysis():
#     return {
#         "비타민": {"current": 75, "target": 100, "status": "양호"},
#         "미네랄": {"current": 60, "target": 100, "status": "부족"},
#         "항산화": {"current": 85, "target": 100, "status": "양호"},
#         "오메가3": {"current": 45, "target": 100, "status": "부족"},
#         "프로바이오틱스": {"current": 30, "target": 100, "status": "부족"}
#     }

# 사이드바 - 개인정보 입력
st.sidebar.header("👤 개인정보 입력")

# 기본 정보
age = st.sidebar.number_input("나이", min_value=1, max_value=120, value=30)
gender = st.sidebar.selectbox("성별", ["남성", "여성"])
# 임신 여부 체크
is_pregnant = None
pregnancy_text = ""  # 기본값

if gender == "여성":
    is_pregnant = st.sidebar.checkbox("임신 여부")
    if is_pregnant:
        pregnancy_text = " (임신 중)"

# 건강 관심사
st.sidebar.subheader("건강 관심사")
health_goals = []
if st.sidebar.checkbox("면역력 강화"):
    health_goals.append("immunity")
if st.sidebar.checkbox("피부 건강"):
    health_goals.append("skin")
if st.sidebar.checkbox("에너지/피로 회복"):
    health_goals.append("energy")
if st.sidebar.checkbox("관절 건강"):
    health_goals.append("joint")
if st.sidebar.checkbox("소화/장 건강"):
    health_goals.append("digest")
if st.sidebar.checkbox("스트레스 관리"):
    health_goals.append("stress")

# # 추가 정보
# allergies = st.sidebar.text_area("알레르기/복용 중인 약물",
#                                  placeholder="예: 갑각류 알레르기, 혈압약 복용 중")

# 세션 상태 초기화
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = []

# 메인 탭 구성
# tab1, tab2, tab3, tab4 = st.tabs(["💬 질의응답", "🎯 맞춤 추천", "🧪 성분 분석", "📊 제품 비교"])
tab1, tab2 = st.tabs(["💬 질의응답", "🎯 맞춤 추천"])

# 탭 1: 질의응답
with tab1:
    st.header("💬 식품의약품안전처 건강기능식품정보 기반 Q&A")

    # 샘플 질문 버튼
    st.subheader("자주 묻는 질문")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🌟 피부에 좋은 영양제는?"):
            user_input = "피부에 좋은 영양제 추천해주세요"
            st.session_state.chat_history.append({"type": "user", "message": user_input})
            with st.spinner("AI가 답변 중입니다..."):
                response = rag_chatbot.run(user_input)
                st.session_state.chat_history.append({"type": "bot", "message": response})

    with col2:
        if st.button("⚡ 피로 회복에 도움되는 것은?"):
            user_input = "피로 회복에 도움이 되는 영양제는 무엇인가요?"
            st.session_state.chat_history.append({"type": "user", "message": user_input})
            with st.spinner("AI가 답변 중입니다..."):
                response = rag_chatbot.run(user_input)
                st.session_state.chat_history.append({"type": "bot", "message": response})

    with col3:
        if st.button("🤔 비타민D와 칼슘 같이 먹어도 되나요?"):
            user_input = "비타민D와 칼슘을 같이 먹어도 되나요?"
            st.session_state.chat_history.append({"type": "user", "message": user_input})
            with st.spinner("AI가 답변 중입니다..."):
                response = rag_chatbot.run(user_input)
                st.session_state.chat_history.append({"type": "bot", "message": response})

    # 채팅 입력
    user_input = st.text_input("궁금한 것을 물어보세요...", key="chat_input")

    if st.button("전송") and user_input:
        st.session_state.chat_history.append({"type": "user", "message": user_input})
        st.success(f"당신의 프로필 : {age}세 {gender}{pregnancy_text}에 맞춘 식품의약품안전처 건강기능식품정보 입니다!")
        
        with st.spinner("AI가 답변 중입니다..."):
            user_input = str(age) + '세 ' + gender + ('(임신중)' if is_pregnant else '') + ' ' + user_input
            response = rag_chatbot.run(user_input)
            st.session_state.chat_history.append({"type": "bot", "message": response})

    # 질문-답변 묶기
    chat_pairs = []
    history = st.session_state.chat_history
    i = 0

    while i < len(history) - 1:
        if history[i]["type"] == "user" and history[i+1]["type"] == "bot":
            chat_pairs.append((history[i], history[i+1]))
            i += 2
        else:
            i += 1  # 짝이 안 맞는 경우 넘어감

    # 최근 것이 위로 오도록 역순 출력
    for user_msg, bot_msg in reversed(chat_pairs):
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>질문:</strong> {user_msg["message"]}
        </div>
        <div class="chat-message bot-message">
            <strong>🤖 NutriWise AI 답변:</strong> {bot_msg["message"]}
        </div>
        """, unsafe_allow_html=True)

# 탭 2: 맞춤 추천
with tab2:
    st.header("🎯 개인 맞춤형 AI 추천")

    if st.button("🔍 맞춤 추천 생성하기", type="primary"):
        if age and gender:
            with st.spinner("AI가 맞춤 추천을 생성하고 있습니다..."):

                # 사용자 조건에 따라 검색 쿼리 설정
                if "energy" in health_goals:
                    query = f"{age}세 {gender}{pregnancy_text}를 위한 피로 회복, 면역력 관련 건강기능식품 추천"
                elif "skin" in health_goals:
                    query = f"{age}세 {gender}{pregnancy_text}를 위한 피부 탄력 관련 영양제 추천"
                elif "digest" in health_goals:
                    query = f"{age}세 {gender}{pregnancy_text}를 위한 장 건강 관련 프로바이오틱스 추천"
                elif "immunity" in health_goals:
                    query = f"{age}세 {gender}{pregnancy_text}를 위한 면역력 강화 영양제 추천"
                elif "joint" in health_goals:
                    query = f"{age}세 {gender}{pregnancy_text}를 위한 관절 건강 영양제 추천"
                elif "stress" in health_goals:
                    query = f"{age}세 {gender}{pregnancy_text}를 위한 스트레스 관리 관련 영양제 추천"
                else:
                    query = f"{age}세 {gender}{pregnancy_text}에게 일반적으로 추천되는 건강기능식품"

                # 웹 기반 추천 호출
                web_result = get_recommendation_from_web(query, cfg)
                try:
                    web_products = json.loads(web_result)
                    st.session_state.recommendations = web_products
                except Exception as e:
                    print("Error parsing web result:", e)
                    st.error("추천 정보를 파싱하는 데 실패했습니다. 응답 내용:\n" + web_result)
                    st.stop()
        else:
            st.error("사이드바에서 개인정보를 모두 입력해주세요.")

    # 결과 렌더링
    if st.session_state.recommendations:
        st.success(f"당신의 프로필({age}세 {gender}{pregnancy_text})에 맞춘 AI 추천 제품입니다!")

        for product in st.session_state.recommendations:
            components.html(f"""
            <style>
                .product-card {{
                    background: #f8fafc;
                    border: 2px solid #e2e8f0;
                    border-radius: 10px;
                    padding: 1.5rem;
                    margin: 1rem 0;
                    border-left: 5px solid #667eea;
                }}
            </style>
                            
            <div class="product-card">
                <h3>🌟 {product['name']}</h3>
                <p><strong>브랜드:</strong> {product['brand']} | <strong>가격:</strong> {product['price']}</p>
                <p><strong>평점:</strong> ⭐ {product['rating']}/5.0 ({product['reviews']}개 리뷰)</p>

                <p><strong>주요 성분:</strong>
                    {' '.join([f'<span class="ingredient-tag">{i}</span>' for i in product['ingredients']])}
                </p>

                <p><strong>기대 효과:</strong>
                    {' • '.join(product['benefits'])}</p>

                <p><strong>복용법:</strong>
                    {product['dosage']}</p>

                <div class="warning-box">
                    <h4>⚠️ 주의사항:</h4>
                    <ul>
                        {''.join([f'<li>{w}</li>' for w in product['warnings']])}
                    </ul>
                </div>
            </div>
            """, height=400)

# - 2차 추후 개발
# # TODO 탭 3: 성분 분석
# with tab3:
#     st.header("🧪 영양 성분 분석 대시보드")

#     nutrition_data = get_nutrition_analysis()

#     # 영양소 현황 메트릭
#     st.subheader("📊 현재 영양소 섭취 현황")
#     cols = st.columns(len(nutrition_data))

#     for i, (nutrient, data) in enumerate(nutrition_data.items()):
#         with cols[i]:
#             delta = data['current'] - data['target']
#             st.metric(
#                 label=nutrient,
#                 value=f"{data['current']}%",
#                 delta=f"{delta:+d}%",
#                 delta_color="normal" if delta >= 0 else "inverse"
#             )

#     # 영양소 차트
#     st.subheader("📈 영양소 섭취 분석")

#     # 레이더 차트
#     categories = list(nutrition_data.keys())
#     current_values = [nutrition_data[cat]['current'] for cat in categories]
#     target_values = [nutrition_data[cat]['target'] for cat in categories]

#     fig = go.Figure()

#     fig.add_trace(go.Scatterpolar(
#         r=current_values,
#         theta=categories,
#         fill='toself',
#         name='현재 섭취량',
#         line_color='rgb(102, 126, 234)'
#     ))

#     fig.add_trace(go.Scatterpolar(
#         r=target_values,
#         theta=categories,
#         fill='toself',
#         name='권장 섭취량',
#         line_color='rgb(255, 99, 132)',
#         opacity=0.6
#     ))

#     fig.update_layout(
#         polar=dict(
#             radialaxis=dict(
#                 visible=True,
#                 range=[0, 100]
#             )),
#         showlegend=True,
#         title="영양소 섭취 현황 비교"
#     )

#     st.plotly_chart(fig, use_container_width=True)

#     # 부족한 영양소 분석
#     st.subheader("⚠️ 부족한 영양소 분석")

#     deficient_nutrients = []
#     for nutrient, data in nutrition_data.items():
#         if data['current'] < 70:  # 70% 미만은 부족으로 판단
#             deficient_nutrients.append({
#                 'nutrient': nutrient,
#                 'current': data['current'],
#                 'gap': 100 - data['current']
#             })

#     if deficient_nutrients:
#         for nutrient in deficient_nutrients:
#             st.warning(f"**{nutrient['nutrient']}** 부족 ({nutrient['current']}% 섭취, {nutrient['gap']}% 부족)")
#     else:
#         st.success("모든 영양소가 권장 섭취량을 만족합니다! 👏")

# # 탭 4: 제품 비교
# with tab4:
#     st.header("📊 제품 비교 분석")

#     products_df = load_sample_data()

#     # 제품 선택
#     selected_products = st.multiselect(
#         "비교할 제품을 선택하세요 (최대 4개)",
#         options=products_df['name'].tolist(),
#         default=products_df['name'].tolist()[:3],
#         max_selections=4
#     )

#     if selected_products:
#         comparison_df = products_df[products_df['name'].isin(selected_products)]

#         # 가격 비교 차트
#         st.subheader("💰 가격 비교")
#         fig_price = px.bar(
#             comparison_df,
#             x='name',
#             y='price',
#             color='category',
#             title="제품별 가격 비교",
#             labels={'price': '가격 (원)', 'name': '제품명'}
#         )
#         st.plotly_chart(fig_price, use_container_width=True)

#         # 평점 비교 차트
#         st.subheader("⭐ 평점 비교")
#         fig_rating = px.scatter(
#             comparison_df,
#             x='rating',
#             y='reviews',
#             size='price',
#             color='category',
#             hover_name='name',
#             title="평점 vs 리뷰 수 (크기: 가격)",
#             labels={'rating': '평점', 'reviews': '리뷰 수'}
#         )
#         st.plotly_chart(fig_rating, use_container_width=True)

#         # 상세 비교 테이블
#         st.subheader("📋 상세 비교")

#         # 테이블 데이터 준비
#         table_data = []
#         for _, product in comparison_df.iterrows():
#             table_data.append({
#                 '제품명': product['name'],
#                 '브랜드': product['brand'],
#                 '가격': f"₩{product['price']:,}",
#                 '평점': f"{product['rating']}/5.0",
#                 '리뷰수': f"{product['reviews']}개",
#                 '주요성분': ', '.join(product['ingredients'][:3]) + '...',
#                 '복용법': product['dosage']
#             })

#         comparison_table = pd.DataFrame(table_data)
#         st.dataframe(comparison_table, use_container_width=True)


# 하단 정보
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>⚠️ 본 서비스는 정보 제공 목적이며, 의료진과 상담 후 복용하시기 바랍니다.</p>
    <p>🔬 AI 추천 시스템은 지속적으로 학습하고 개선됩니다.</p>
</div>
""", unsafe_allow_html=True)

# TODO 실제 구현 시 필요한 추가 기능들
# 실제 구현 시 추가할 기능들:
#
# 1. 데이터베이스 연동:
#    - PostgreSQL/MongoDB에 제품 정보 저장
#    - 사용자 프로필 및 추천 기록 저장
#
# 2. LLM + RAG 시스템:
#    - OpenAI API 모델 사용
#    - 벡터 데이터베이스 (Pinecone) 연동
#    - 제품 문서 임베딩 및 검색
#
# 3. 실시간 데이터 수집:
#    - iHerb, 쿠팡 등 쇼핑몰 크롤링
#    - 가격 변동 추적
#    - 리뷰 감성 분석
#
# 4. 개인화 기능:
#    - 사용자 로그인/회원가입
#    - 복용 기록 추적
#    - 알림 시스템
#
# 5. 고급 분석:
#    - 영양소 상호작용 분석
#    - 개인 건강 데이터 연동
#    - 의료진 상담 예약 시스템

