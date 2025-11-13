import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm.openai import OpenAI
import os

# --- LLM 클래스 지정 및 Streamlit Cloud Secrets에서 API 키 로드 ---
def get_api_key():
    """Streamlit Secrets 또는 환경 변수에서 API 키를 가져옵니다."""
    try:
        # 1. Streamlit Cloud Secrets에서 키를 시도합니다. (배포 환경)
        return st.secrets["OPENAI_API_KEY"]
    except KeyError:
        # 2. 로컬 테스트를 위해 환경 변수에서 키를 시도합니다.
        return os.environ.get("OPENAI_API_KEY")

api_key = get_api_key()

# 1. 페이지 설정
st.set_page_config(page_title="📊 PandasAI 기반 엑셀 분석기", layout="centered")
st.title("📊 GPT-3.5 Turbo 기반 데이터 분석기")
st.markdown("Streamlit Cloud Secrets를 사용하여 API 키 노출 없이 안전하게 분석을 수행합니다. (키 이름: `OPENAI_API_KEY`)")

if not api_key:
    st.error("❌ 오류: Streamlit Cloud Secrets나 로컬 환경 변수 **'OPENAI_API_KEY'**가 설정되지 않았습니다.")
    st.info("앱을 실행하려면, 해당 환경 변수에 실제 API 키를 설정해야 합니다.")
else:
    # 2. 파일 업로드 및 데이터 로드
    uploaded_file = st.file_uploader(
        "1. 분석할 엑셀 파일(.xlsx)을 업로드하세요.",
        type=["xlsx"],
        help="PandasAI는 파일 업로드 후 데이터를 GPT에 전송하여 분석합니다."
    )

    if uploaded_file is not None:
        st.success("✅ 파일 업로드 완료. 데이터를 로드합니다.")
        
        try:
            # 엑셀 파일 로드
            data = pd.read_excel(uploaded_file)
            st.subheader("2. 업로드된 데이터 미리보기")
            st.dataframe(data.head()) # 상위 5행 표시
            st.info(f"데이터 크기: {data.shape[0]} 행, {data.shape[1]} 열")
            
            # 3. LLM 연결 및 SmartDataframe 초기화
            with st.spinner("⏳ LLM 연결 및 SmartDataframe 초기화 중..."):
                
                # OpenAI LLM 초기화 (V3.0.0 호환성을 위해 api_key 사용)
                llm = OpenAI(model="gpt-3.5-turbo", api_key=api_key)
                
                # SmartDataframe 초기화 (config 딕셔너리 방식)
                sdf = SmartDataframe(data, config={"llm": llm})
                
                st.success("✅ LLM 및 SmartDataframe 초기화 성공!")

            # 4. 사용자 입력 및 분석 실행
            st.subheader("3. 분석 질문 입력")
            
            # Form을 사용하여 입력과 버튼 클릭을 명확하게 분리
            with st.form("analysis_form"):
                user_prompt = st.text_area(
                    "데이터에 대해 알고 싶은 내용을 질문하세요:",
                    placeholder="예: 'Floor 별 사전 제작 물량의 합계를 표로 보여줘'",
                    key="user_prompt"
                )
                submitted = st.form_submit_button("AI 분석 실행")
            
            if submitted:
                if user_prompt.strip():
                    with st.spinner("⏳ GPT-3.5 Turbo가 분석 코드를 생성하고 실행 중입니다..."):
                        try:
                            # PandasAI 질의 수행
                            result = sdf.chat(user_prompt)
                            
                            st.subheader("💡 분석 결과")
                            
                            # 결과 출력: DataFrame 또는 단순 문자열/값
                            if isinstance(result, pd.DataFrame):
                                st.dataframe(result)
                            else:
                                st.write(result)
                                
                            st.success("✅ 분석 완료!")

                        except Exception as e:
                            st.error(f"❌ 분석 중 오류 발생: {e}")
                            st.warning("질문 내용이 모호하거나 데이터 형식에 문제가 있을 수 있습니다. 질문을 구체화하거나 데이터 형식을 확인해 주세요.")
                else:
                    st.warning("분석 질문을 입력해 주세요.")
                    
        except Exception as e:
            st.error(f"❌ 데이터 로드 오류: 파일 내용이나 형식을 확인해 주세요. ({e})")
```eof
