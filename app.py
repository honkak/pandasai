# ======================================================
# 본격 편집 중 REV7 - PandasAI Streamlit App (CTk 로직 완전 이식)
# ======================================================

import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm.openai import OpenAI
import openai
from typing import Optional, Any, Dict, List, Tuple
import re
import sys
import json
import os

# ======================================================
# 0. 설정 및 상수 정의
# ======================================================
LLM_MODEL = "gpt-3.5-turbo"  # "gpt-3.5-turbo", "gpt-4o"
RESET_ON_QUERY = True  # True: 매 쿼리마다 SmartDataframe 재생성 / False: 세션 재사용

# ======================================================
# 📌 LLM 동작 규칙 (원본 그대로)
# ======================================================
CUSTOM_INSTRUCTION = """
이 데이터프레임의 분석을 위해 반드시 다음 규칙을 따르세요.

========================================================
1. DataFrame 사용 규칙
========================================================
- SmartDataframe 내부의 df '하나만' 사용해야 합니다.
- df 외에 dfs, temp_df, new_df 등 새로운 리스트나 데이터프레임을 만들지 마십시오.
- 절대 df를 리스트로 감싸거나 반복문으로 처리하지 마십시오.

========================================================
2. 필터링 규칙 (핵심)
========================================================
DataFrame 필터링은 반드시 아래 형식만 허용합니다:

df_filtered = df[
    (df['컬럼'] == 값) &
    (df['컬럼'] == 값)
]

아래 동작은 절대 금지합니다:
- (df['컬럼'] == 값).all()
- for df in dfs
- dfs = [df for df in dfs ...]
- pd.concat()
- 여러 개의 df를 리스트에 담아 처리

========================================================
3. 그룹바이/집계 규칙
========================================================
- 집계(sum, mean 등)는 단일 df 객체에서만 수행하십시오.
- df.groupby(...) 는 허용됩니다.
- df_list, concat, merge 등 두 개 이상의 DF를 만들어 조작하는 행위를 금지합니다.

========================================================
4. 결과 반환 규칙
========================================================
반드시 아래 형식으로 반환해야 합니다:

result = {"type": "dataframe", "value": df_filtered}

========================================================
5. 코드 안전 규칙
========================================================
- Python 문법 오류가 발생하는 코드는 생성하지 마십시오.
- 존재하지 않는 변수(dfs, temp_df 등)를 사용하지 마십시오.
"""

# ======================================================
# 컬럼 및 값 동의어 (원본 그대로)
# ======================================================
COLUMN_SYNONYMS = {
    "장비명": ["장비"],
    "UT": ["공종", "설비", "유틸리티", "utility"],
    "Floor": ["층수", "플로어"],
    "사전제작X_비대상(일부공정)(1)_길이": ["비대상"],
    "사전제작X_A(장비단Final)(2)_길이": ["장비단"],
    "사전제작○_B(H_UP구간)(3)_당초계획_길이": ["계획물량"],
    "사전제작○_B(H_UP구간)(4)_실제시공_길이": ["시공물량"],
    "사전제작X_C(TV단Final)(5)_길이": ["테핑밸브단"],
    "합계(1+2+4+5)": ["총합"]
}

VALUE_SYNONYMS = {
    "1F": ["1층"],
    "2F": ["2층"],
    "3F": ["3층"],
    "Bulk Gas": ["벌크가스", "bulk gas"],
    "Drain": ["드레인", "drain"],
    "Exhaust": ["이그저스트", "exhaust"],
    "UPW(DI)": ["초순수"],
    "PCW": ["프로세스쿨링워터"],
    "NPW": ["공업용수"],
    "Chemical": ["케미칼", "chemical"],
    "Pumping": ["펌프", "pumping"],
    "Toxic Gas": ["톡식가스", "toxic gas"],
}

# ======================================================
# Streamlit 페이지 설정
# ======================================================
st.set_page_config(
    page_title="📊 PandasAI 대화형 데이터 분석기 (Streamlit)",
    layout="wide"
)

st.sidebar.subheader("🔐 OpenAI API 키 입력")
if "OPENAI_API_KEY" not in st.session_state:
    st.session_state.OPENAI_API_KEY = ""


api_key_input = st.sidebar.text_input("OpenAI API Key 입력 (sk-...)", type="password", value=st.session_state.OPENAI_API_KEY)
if st.sidebar.button("💾 키 저장"):
    if api_key_input.startswith("sk-"):
        st.session_state.OPENAI_API_KEY = api_key_input
        st.sidebar.success("✅ API 키 저장 완료")
    else:
        st.sidebar.warning("⚠️ 유효한 OpenAI 키 형식이 아닙니다.")

# ======================================================
# API 키 확인 후 진행
# ======================================================
if not st.session_state.get("OPENAI_API_KEY", "").startswith("sk-"):
    st.warning("👈 왼쪽에서 OpenAI API 키를 먼저 입력하고 저장하세요.")
    st.stop()


# ======================================================
# 1. 분석 환경 초기화 (AnalysisInitializer) - 폴더 순회 로직 유지/응용
# ======================================================
class AnalysisInitializer:
    def __init__(self, uploaded_file):
        self._model = LLM_MODEL
        self._instruction = CUSTOM_INSTRUCTION
        self.uploaded_files = uploaded_files   # ✅ 여러 파일 지원
        self.llm: Optional[OpenAI] = None
        self.sdf: Optional[SmartDataframe] = None

    def initialize(self) -> Tuple[SmartDataframe, pd.DataFrame, OpenAI]:
        api_key = st.session_state["OPENAI_API_KEY"]
        openai.api_key = api_key

        df = self._load_data()

        # 👈 LLM 인스턴스를 여기서 생성
        self.llm = OpenAI(api_token=api_key, model=self._model)

        # ★ PandasAI v2.3.2 : df 그대로 전달
        self.sdf = SmartDataframe(
            df,
            config={
                "llm": self.llm,
                "verbose": True,
                "memory": False,
                "instructions": CUSTOM_INSTRUCTION
            }
        )

        return self.sdf, df, self.llm

    # ======================================================
    # 엑셀 로드 → 전처리 → 여러 개 업로드된 파일 병합
    # ======================================================
    def _load_data(self) -> pd.DataFrame:
        if not self.uploaded_files:
            raise FileNotFoundError("⚠️ 업로드된 엑셀 파일이 없습니다.")

        excel_files = self.uploaded_files  # ✅ 여러 파일 직접 사용

        print(f"📂 총 {len(excel_files)}개 파일 감지됨:")
        for f in excel_files:
            print(f" - {getattr(f, 'name', 'uploaded_file')}")

        all_dfs = []
        
        # --------------------------------------------------
        # 1️⃣ 개별 파일 전처리 (원본과 동일한 로직)
        # --------------------------------------------------
        for file in excel_files:
            file_name = getattr(file, "name", "uploaded_file")
            print(f"🔄 전처리 중: {file_name}")
            try:
                df_raw = pd.read_excel(file, header=None)

                # 상단 4줄 삭제
                df_raw = df_raw.iloc[4:].reset_index(drop=True)

                # 불필요한 열 제거 (E, G, I, K, L, N, O)
                drop_cols = [4, 6, 8, 10, 11, 13, 14]
                df_raw = df_raw.drop(df_raw.columns[drop_cols], axis=1)

                # 새 헤더 지정
                new_columns = [
                    "장비명", "UT", "Floor",
                    "사전제작X_비대상(일부공정)(1)_길이",
                    "사전제작X_A(장비단Final)(2)_길이",
                    "사전제작○_B(H_UP구간)(3)_당초계획_길이",
                    "사전제작○_B(H_UP구간)(4)_실제시공_길이",
                    "사전제작X_C(TV단Final)(5)_길이"
                ]
                df_raw.columns = new_columns

                # 숫자형 변환
                for col in new_columns[3:]:
                    df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")

                # 합계(1+2+4+5) 계산
                df_raw["합계(1+2+4+5)"] = pd.to_numeric(
                    df_raw["사전제작X_비대상(일부공정)(1)_길이"].fillna(0)
                    + df_raw["사전제작X_A(장비단Final)(2)_길이"].fillna(0)
                    + df_raw["사전제작○_B(H_UP구간)(4)_실제시공_길이"].fillna(0)
                    + df_raw["사전제작X_C(TV단Final)(5)_길이"].fillna(0),
                    errors="coerce"
                ).astype("float64")

                all_dfs.append(df_raw)
                print(f"✅ {file_name} 전처리 완료: {len(df_raw)}행")

            except Exception as e:
                print(f"❌ {file_name} 처리 중 오류 발생: {e}")

        # --------------------------------------------------
        # 2️⃣ 병합 (원본 구조 유지)
        # --------------------------------------------------
        if not all_dfs:
            raise RuntimeError("❌ 전처리에 성공한 파일이 없습니다.")

        for df_raw in all_dfs:
            if "합계(1+2+4+5)" in df_raw.columns:
                df_raw["합계(1+2+4+5)"] = pd.to_numeric(
                    df_raw["합계(1+2+4+5)"], errors="coerce"
                )

        merged_df = pd.concat(all_dfs, ignore_index=True)
        print(f"\n📊 전체 병합 완료: 총 {len(merged_df)}행, {len(merged_df.columns)}열")

        return merged_df


# ======================================================
# 2. 질문 가공 로직 (최종 안정 버전 그대로)
# ======================================================
class PromptPreprocessor:
    def __init__(self):
        self._column_synonyms = COLUMN_SYNONYMS
        self._value_synonyms = VALUE_SYNONYMS
        self._ut_exclude = ["장비", "장비들"]

        self._dimension_ut_words = ["UT", "공종", "설비", "유틸리티", "utility"]
        self._dimension_device_words = ["장비명", "장비"]
        self._dimension_floor_words = ["층", "층수"]

        # ✅ 한글 조사 리스트 (필요하면 더 추가해도 됨)
        self._josa_list = [
            "은", "는", "이", "가",
            "을", "를", "의",
            "에", "에서",
            "로", "으로",
            "와", "과",
            "도"
        ]

    def _normalize_column_words(self, prompt: str) -> str:
        """컬럼 동의어 + 뒤에 붙은 조사까지 인식해서 컬럼명을 정규화"""

        # 조사 패턴: 위 리스트 중 1개 또는 2글자짜리 조사도 있으니 전체 OR
        josa_pattern = "(?:" + "|".join(self._josa_list) + ")?"

        for target, syns in self._column_synonyms.items():
            # syns(별칭) + target(정규 컬럼명) 둘 다 잡도록
            for syn in syns + [target]:
                # ✅ 한글이 들어간 동의어인 경우: 우리가 직접 경계 정의 + 조사 허용
                if re.search(r"[가-힣]", syn):
                    pattern = rf"(?<![가-힣A-Za-z0-9])" \
                              rf"({re.escape(syn)})" \
                              rf"{josa_pattern}" \
                              rf"(?=[^가-힣A-Za-z0-9]|$)"
                    prompt = re.sub(pattern, target, prompt)
                else:
                    # ✅ 영문/숫자 위주의 동의어(utility 등)는 기존 \b 로 그대로 처리
                    pattern = rf"\b{re.escape(syn)}\b"
                    prompt = re.sub(pattern, target, prompt, flags=re.IGNORECASE)

        return prompt

    # ======================================================
    # 메인 처리 함수 (원본 로직 그대로)
    # ======================================================
    def process(self, raw_prompt: str) -> str:
        if not raw_prompt:
            return ""

        prompt = raw_prompt.strip()
        conditions = []
        selected_columns = []
        dimension_columns = []

        # --------------------------------------------
        # 1. 기존 컬럼/값 동의어 치환
        # --------------------------------------------

        # 🚀 '장비'를 'equipment'로 치환하여 컬럼 동의어 충돌 방지
        prompt = re.sub(r"\b장비\b", "equipment", prompt)

        # ✅ 컬럼명/별칭 + 조사까지 포함해서 정규화
        prompt = self._normalize_column_words(prompt)

        for target, syns in self._value_synonyms.items():
            for syn in syns:
                prompt = re.sub(
                    rf"\b{re.escape(syn)}\b", target, prompt, flags=re.IGNORECASE
                )

        prompt = re.sub(r"\b배관\b", "유틸리티", prompt)
        prompt = re.sub(r"\b물량\b", "물량들", prompt)

        # ----------------------------------------------------
        # 2. ⭐ 차원 분석(별, 띄어쓰기 모두 감지)
        # ----------------------------------------------------

        for word in self._dimension_ut_words:
            if re.search(rf"{word}\s*별", raw_prompt, flags=re.IGNORECASE):
                dimension_columns.append("UT")
                prompt = re.sub(rf"{word}\s*별", "", prompt)
                break

        for word in self._dimension_device_words:
            if re.search(rf"{word}\s*별", raw_prompt, flags=re.IGNORECASE):
                dimension_columns.append("장비명")
                prompt = re.sub(rf"{word}\s*별", "", prompt)
                break

        for word in self._dimension_floor_words:
            if re.search(rf"{word}\s*별", raw_prompt, flags=re.IGNORECASE):
                dimension_columns.append("Floor")
                prompt = re.sub(rf"{word}\s*별", "", prompt)
                break

        # ----------------------------------------------------
        # 3. 기본 조건(Floor/UT/장비명 감지)
        # ----------------------------------------------------

        for fl in ["1F", "2F", "3F"]:
            if re.search(rf"\b{fl}\b", prompt):
                conditions.append(f'(Floor == "{fl}")')
                prompt = re.sub(rf"\b{fl}\b", "", prompt)

        for val in self._value_synonyms.keys():
            if val not in ["1F", "2F", "3F"] and val not in self._ut_exclude:
                if re.search(rf"\b{val}\b", prompt):
                    conditions.append(f'(UT == "{val}")')
                    prompt = re.sub(rf"\b{val}\b", "", prompt)

        device_matches = [
            word for word in re.findall(r"\b[A-Za-z0-9]{3,}\b", prompt)
            if any(c.isalpha() for c in word) and any(c.isdigit() for c in word)
        ]

        for dev in device_matches:
            conditions.append(f'(장비명 == "{dev}")')
            prompt = prompt.replace(dev, "")

        # ----------------------------------------------------
        # 4. 출력 컬럼 자동 감지
        # ----------------------------------------------------

        for col in self._column_synonyms.keys():
            if col in prompt:
                selected_columns.append(col)
                prompt = prompt.replace(col, "")

        # ----------------------------------------------------
        # 5. 명령어 표준화 (단순화 로직 적용)
        # ----------------------------------------------------

        # 띄어쓰기를 제외한 문자열에서 한글만 추출
        korean_chars = re.sub(r"[^가-힣]", "", prompt)

        # 한글이 2글자 이상 포함되어 있다면 표준 명령 삽입
        if len(korean_chars) >= 2:

            # 기존에 있던 '보여줘/알려줘/구해줘' 등의 패턴을 먼저 제거합니다.
            command_patterns = r"(보여줘|알려줘|구해줘|리스트해줘|정리해줘|목록화해줘|합은|총합은|합계는|총량은|몇이야|몇개야|얼마야|어떻게 돼|얼마인지|결과는)"
            prompt = re.sub(command_patterns, "", prompt)

            # 새로운 표준 명령 삽입
            prompt = prompt.strip() + " 데이터프레임으로 보여줘"

        # ----------------------------------------------------
        # 6. 조립
        # ----------------------------------------------------

        final_parts = []
        if conditions:
            final_parts.append(" AND ".join(conditions))
        if selected_columns:
            final_parts.append(f"출력컬럼 = {['장비명','UT','Floor'] + selected_columns}")
        if dimension_columns:
            final_parts.append(f"차원컬럼 = {dimension_columns}")
            final_parts.append("집계방식 = 'sum'")
        final_parts.append(prompt)
        final = " ".join(final_parts)
        final = re.sub(r"\s+", " ", final).strip()

        return final


# ======================================================
# 스마트 응답 모듈 (Smart Response Engine) - 원본 그대로
# ======================================================
class SmartResponseEngine:
    def __init__(self):
        pass

    # 1. 결과가 DF인지 확인
    def is_dataframe(self, result: Any) -> bool:
        if isinstance(result, dict) and result.get("type") == "dataframe":
            return True
        if isinstance(result, pd.DataFrame):
            return True
        return False

    # 2. DF 분석 (SUM/MEAN/MAX/MIN 계산)
    def analyze_dataframe(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        df = df.apply(pd.to_numeric, errors="ignore")
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
        stats = {}

        for col in numeric_cols:
            stats[col] = {
                "sum": float(df[col].sum()),
                "mean": float(df[col].mean()),
                "max": float(df[col].max()),
                "min": float(df[col].min())
            }

        return stats

    # 3. 스마트 응답을 DataFrame 형태로 생성 (llm 인스턴스 추가)
    def generate_smart_response(self, df_stats: Dict, prompt: str, llm_instance: OpenAI) -> Tuple[str, pd.DataFrame]:
        # ▶️ 1. 통계표 생성용 데이터
        stats_dict = {
            "SUM": {},
            "MEAN": {},
            "MAX": {},
            "MIN": {}
        }

        def format_value(v: float) -> str:
            rounded = round(v, 2)
            return str(int(rounded)) if rounded == int(rounded) else f"{rounded:.2f}"

        # ✅ 합계(1+2+4+5) 컬럼의 sum을 기준 total_sum으로 사용
        total_sum_entry = df_stats.get("합계(1+2+4+5)")
        total_sum = float(total_sum_entry["sum"]) if total_sum_entry else 0.0

        # ▶️ 2. JSON 구조 생성 (LLM 전달용)
        stats_json = {}
        for col, values in df_stats.items():
            entry = {"sum": round(values["sum"], 2)}

            # ✅ total_sum 비율 계산
            if total_sum > 0:
                entry["ratio_to_total"] = round(values["sum"] / total_sum * 100, 1)

            # ✅ 계획 대비 시공 비율 계산
            if "당초계획" in col:
                for other_col in df_stats:
                    if "실제시공" in other_col:
                        plan = values["sum"]
                        real = df_stats[other_col]["sum"]
                        if plan > 0:
                            ratio = round(real / plan * 100, 1)
                            diff = round(real - plan, 2)
                            trend = (
                                "시공이 계획보다 많음" if real > plan
                                else "시공이 계획보다 적음" if real < plan
                                else "계획과 동일"
                            )
                            stats_json[other_col] = stats_json.get(other_col, {})
                            stats_json[other_col].update({
                                "plan_to_real_ratio": ratio,
                                "plan_to_real_diff": diff,
                                "plan_to_real_trend": trend
                            })

            stats_json[col] = entry

            # ✅ 합계 컬럼은 SUM만 표시, 나머지는 '-'
            if col == "합계(1+2+4+5)":
                stats_dict["SUM"][col] = format_value(values["sum"])
                stats_dict["MEAN"][col] = "-"
                stats_dict["MAX"][col] = "-"
                stats_dict["MIN"][col] = "-"
            else:
                stats_dict["SUM"][col] = format_value(values["sum"])
                stats_dict["MEAN"][col] = format_value(values["mean"])
                stats_dict["MAX"][col] = format_value(values["max"])
                stats_dict["MIN"][col] = format_value(values["min"])

        # ✅ 사전제작 물량 vs 비진행 물량 비교 
        try:
            pre_fab_sum = df_stats.get("사전제작○_B(H_UP구간)(4)_실제시공_길이", {}).get("sum", 0)
            non_pre_fab_sum = sum([
                df_stats.get("사전제작X_비대상(일부공정)(1)_길이", {}).get("sum", 0),
                df_stats.get("사전제작X_A(장비단Final)(2)_길이", {}).get("sum", 0),
                df_stats.get("사전제작X_C(TV단Final)(5)_길이", {}).get("sum", 0)
            ])

            if non_pre_fab_sum > 0:
                pre_ratio = round(pre_fab_sum / non_pre_fab_sum * 100, 1)

                if pre_ratio > 100:
                    trend = f"사전제작 진행물량은 비진행 물량보다 {round(pre_ratio / 100, 2)}배 많습니다."
                elif pre_ratio == 100:
                    trend = "사전제작 진행물량과 비진행 물량은 동일한 수준입니다."
                else:
                    trend = f"사전제작 진행물량은 비진행 물량 대비 {pre_ratio}% 수준으로 상대적으로 적습니다."

                stats_json["사전제작_비진행_비교"] = {
                    "사전제작_물량합계": round(pre_fab_sum, 2),
                    "비진행_물량합계": round(non_pre_fab_sum, 2),
                    "사전제작_비율(비진행_기준%)": pre_ratio,
                    "비교결과": trend
                }

        except Exception as e:
            stats_json["사전제작_비진행_비교"] = {"오류": str(e)}

        stats_df = pd.DataFrame(stats_dict)

        stats_json_str = json.dumps(stats_json, ensure_ascii=False, indent=2)

        # ▶️ 3. LLM 프롬프트 구성
        insight_prompt = f"""
다음은 특정 설비 배관 데이터에 대한 정량 분석 결과이다.
주어진 수치를 참고하여 현장 엔지니어 관점에서 의미 있는 인사이트를 5문장 이내로 생성하라.

단:
- 어떤 항목을 강조할지 스스로 판단하라.
- 비율 및 변화량은 반드시 JSON에 제공된 숫자만 사용한다.
- '계획 대비 시공', '합계 대비 비율', '가장 큰 항목', '사전제작 진행물량' 등은 필요 시 선택적으로 언급하라.
- '사전제작 vs 비진행 비교'는 반드시 "사전제작은 비진행 대비 xx% 수준"으로 표현하라.
- 사전제작이 적은 경우 '작게 나타났다' 또는 '상대적으로 적다' 등의 표현을 사용할 것.
- 인사이트는 '데이터를 해석한 문장'이어야 하며, 다시 숫자를 나열하지 마라.

JSON 데이터:
{{
  "stats": {stats_json_str},
  "total_sum": {round(total_sum, 2)}
}}
        """.strip()

        # ▶️ 4. LLM 호출 (직접 OpenAI SDK 사용)
        try:
            from openai import OpenAI as OpenAIClient
            client = OpenAIClient(api_key=openai.api_key)

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a senior data analysis assistant."},
                    {"role": "user", "content": insight_prompt}
                ],
                temperature=0.7,
                max_tokens=400
            )

            insight = response.choices[0].message.content.strip()

        except Exception as e:
            insight = f"⚠️ 인사이트 생성 실패: {e}"

        # ▶️ 5. 질문 요약 + 인사이트 조합
        cleaned_prompt = self.clean_prompt_for_summary(prompt)
        summary_text = (
            "📌 **AI 스마트 분석 결과**\n\n"
            f"💬 분석 요청 요약: **{cleaned_prompt}**\n\n"
            f"🧠 **LLM 인사이트 요약:**\n\n{insight.strip()}\n"
        )

        return summary_text, stats_df

    # 4. 스마트 자연어 응답에서 데이터 프레임 문구 제거
    def clean_prompt_for_summary(self, prompt: str) -> str:
        """
        자연어 응답용 질문 정제: 
        '데이터프레임으로 보여줘' → '데이터 기반으로 AI 분석을 통해 인사이트와 활용방안을 제공합니다.'
        """
        # 제거할 명령어 패턴
        remove_patterns = [
            r"데이터프레임으로\s*보여줘",
            r"보여줘",
            r"데이터프레임",
            r"알려줘",
            r"구해줘",
            r"리스트해줘",
            r"정리해줘",
            r"목록화해줘",
            r"결과는",
            r"합은",
            r"총합은"
        ]

        clean = prompt

        # 불필요 명령 제거
        for p in remove_patterns:
            clean = re.sub(p, "", clean).strip()

        # 마지막 문구를 고정해서 붙여줌
        clean += " — 데이터 기반으로 AI 분석을 통해 인사이트와 활용방안을 제공합니다."

        return clean


# ======================================================
# 4. Streamlit UI (CTk App.perform_analysis 로직을 그대로 옮김)
# ======================================================

st.title("📊 PandasAI 대화형 데이터 분석기")
st.markdown("---")

# --- 사이드바: 엑셀 업로드 ---
st.sidebar.header("📁 엑셀 업로드")
uploaded_files = st.sidebar.file_uploader(
    "사전배관제작 물량 엑셀 파일을 선택하세요 (.xlsx)",
    type=["xlsx"],
    accept_multiple_files=True  # ✅ 여러 개 파일 허용
)

if not uploaded_files:
    st.info("👈 왼쪽 사이드바에서 엑셀 파일을 업로드하면 분석을 시작할 수 있습니다.")
    st.stop()
    
initializer = AnalysisInitializer(uploaded_files)  # 리스트 그대로 전달

# --- 초기화 (RESET_ON_QUERY 고려해서 세션에 저장) ---
if "sdf" not in st.session_state or "df" not in st.session_state or "llm" not in st.session_state or RESET_ON_QUERY:
    sdf_instance, df, llm_instance = initializer.initialize()
    st.session_state.sdf = sdf_instance
    st.session_state.df = df
    st.session_state.llm = llm_instance
else:
    sdf_instance = st.session_state.sdf
    df = st.session_state.df
    llm_instance = st.session_state.llm

preprocessor = PromptPreprocessor()
engine = SmartResponseEngine()

st.markdown("## 💬 분석 질문 입력")

with st.form("query_form"):
    user_query = st.text_input(
        "분석할 내용을 입력하고 Enter 또는 버튼을 눌러 실행하세요.",
        placeholder="예: 5TFSP1001 2층 톡식가스 물량 알려줘"
    )
    submitted = st.form_submit_button("🚀 AI 분석 실행")

if submitted:
    if not user_query.strip():
        st.warning("분석 질문을 입력해주세요.")
    else:
        with st.spinner("⏳ AI가 분석 중입니다..."):
            # 1) 질문 가공
            processed = preprocessor.process(user_query)

            # 2) PandasAI 실행 (CTk의 perform_analysis 로직 대응)
            try:
                result = sdf_instance.chat(processed)
                generated_code = sdf_instance.last_code_generated
            except Exception as e:
                st.error(f"❌ 분석 오류: {e}")
                st.stop()

            # 3) 상단: 질문 가공 결과
            st.markdown("### ✨ 질문 가공 결과")
            st.code(processed, language="text")

            # 4) 중간: LLM 생성 코드
            st.markdown("### 💻 LLM 생성 코드")
            st.code(generated_code, language="python")

            # 5) 하단: AI 분석 결과 + 스마트 통계 요약
            st.markdown("### 💡 AI 분석 결과")

            if engine.is_dataframe(result):
                df_out = result.get("value", result)

                # ✅ 이 한 줄로 실제 필터링된 df를 화면에 표시
                st.subheader("📋 필터링된 데이터프레임 결과")
                st.dataframe(df_out)
                
                # 통계 분석
                stats = engine.analyze_dataframe(df_out)

                # 스마트 응답 생성
                summary_text, smart_df = engine.generate_smart_response(
                    stats, processed, llm_instance
                )

                # 요약 텍스트
                st.markdown(summary_text)

                # 통계 DF 출력
                st.markdown("#### 📊 [AI 스마트 통계 요약]")
                st.dataframe(smart_df)
            else:
                # result가 DF가 아니라면 그대로 출력
                st.write(result)











