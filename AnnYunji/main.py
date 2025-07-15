import streamlit as st
from rag import get_rag_response

st.set_page_config(page_title="건강기능식품 상담", layout="wide")

st.title("건강기능식품 추천")
st.markdown("#### 고민이나 증상을 입력하면 적절한 건강기능식품을 추천해드려요.")

query = st.text_input("건강 고민이나 증상을 입력해주세요.", placeholder="예: 요즘 피곤하고 눈이 침침해요")

if st.button("추천받기") and query.strip():
    with st.spinner("추천 중입니다..."):
        try:
            response = get_rag_response(query)
            st.markdown("### 추천 결과")
            st.write(response)
        except Exception as e:
            st.error(f"오류 발생: {e}")
