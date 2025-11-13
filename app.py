import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm.openai import OpenAI
import openai

# ======================================================
# Streamlit 환경 설정
# ======================================================
st.set_page_config(
    page_title="PandasAI 대화형 데이터 분석기",
    layout="wide"
)

st.title("📊 LLM 기반 데이터 분석기 (PandasAI + OpenAI)")
st.caption("엑셀 파일을 업로드하고, 데이터에 대해 질문해보세요.")

# LLM 모델 지정
LLM_MODEL = "gpt-3.5-turbo" 

# ======================================================
# 1. API 키 로드 (Streamlit Secrets 사용)
# ======================================================
api_key = st.secrets.get("OPENAI_API_KEY")

if not api_key:
    st.error("❌ 오류: Streamlit 시크릿에 'OPENAI_API_KEY'가 설정되어 있지 않습니다.")
    st.stop()
    
# OpenAI SDK의 전역 API 키 설정 (선택적이지만 일관성을 위해 유지)
openai.api_key = api_key
st.sidebar.success("✅ OpenAI API 키 로드 성공 (Streamlit Secrets)")

# ======================================================
# 2. 엑셀 데이터 로드 (파일 업로드 사용)
# ======================================================
uploaded_file = st.file_uploader(
    "1. 분석할 엑셀 파일(.xlsx)을 업로드하세요.",
    type=["xlsx"]
)

if uploaded_file is not None:
    try:
        # 업로드된 파일을 Pandas DataFrame으로 로드합니다.
        df = pd.read_excel(uploaded_file)
        st.success(f"✅ 데이터 로드 성공: {uploaded_file.name} (크기: {df.shape})")
        
        st.subheader("데이터 미리보기")
        st.dataframe(df.head())
        
        # ======================================================
        # 3. PandasAI용 LLM 객체 생성 및 SmartDataframe 초기화
        # ======================================================
        @st.cache_resource
        def initialize_pandasai(dataframe, key, model):
            # PandasAI용 OpenAI LLM 객체 생성
            llm_obj = OpenAI(api_token=key, model=model)
            
            # SmartDataframe 초기화
            sdf = SmartDataframe(
                dataframe, 
                config={"llm": llm_obj, "verbose": True, "enable_cache": False}
            )
            return sdf

        sdf = initialize_pandasai(df, api_key, LLM_MODEL)
        st.sidebar.success(f"✅ SmartDataframe 초기화 성공 (모델: {LLM_MODEL})")

        # ======================================================
        # 4. 사용자 질문 처리 및 AI 분석 실행
        # ======================================================
        st.subheader("2. 분석 질문 입력")
        user_prompt = st.text_input(
            "데이터에 대해 알고 싶은 것을 질문하세요 (예: '장비별 총 물량의 합계는?')"
        ).strip()
        
        if user_prompt:
            st.info("⏳ AI 분석 중... (Pandas 코드를 생성하고 실행합니다.)")
            
            try:
                # SmartDataframe의 chat 메서드를 호출하여 분석을 수행합니다.
                with st.spinner("GPT-3.5 Turbo가 분석 코드를 생성하고 있습니다..."):
                    result = sdf.chat(user_prompt)
                
                st.subheader("\n💡 AI 분석 결과")
                # 결과가 DataFrame일 경우 Streamlit의 dataframe으로 표시
                if isinstance(result, pd.DataFrame):
                    st.dataframe(result)
                else:
                    # 결과가 문자열(설명, 숫자 등)일 경우 Markdown으로 표시
                    st.markdown(f"**{result}**")
                
                # verbose=True로 인해 생성된 코드를 확인하기 위해 로그 출력
                st.markdown("---")
                st.markdown("**PandasAI가 생성한 코드 (디버그)**")
                st.code(sdf.last_code_generated, language='python')
                
            except Exception as e:
                st.error(f"❌ 분석 중 오류 발생: {e}")
                st.warning("🚨 참고: OpenAI API 관련 오류(RateLimitError, BillingError 등)일 경우, 계정의 결제 상태를 확인해주세요.")

    except Exception as e:
        st.error(f"❌ 파일 처리 오류: 엑셀 파일이 유효한 형식이 아닙니다. ({e})")

else:
    st.info("⬆️ 분석을 시작하려면 엑셀 파일을 업로드해주세요.")
