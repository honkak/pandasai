########################################
#주식 차트비교 서비스 개발_2025.10.06#
########################################

import streamlit as st
import FinanceDataReader as fdr
import datetime
import pandas as pd
import yfinance as yf
# import streamlit_analytics <- 제거됨

#서비스 제목 입력
st.markdown("<h2 style='font-size: 24px; text-align: center;'>다빈치 주식차트 겹치기 </h2>", unsafe_allow_html=True)

# 날짜 입력 (조회 시작일과 종료일을 같은 행에 배치)
col_start_date, col_end_date = st.columns(2)

with col_start_date:
    start_date = st.date_input(
        "조회 시작일을 선택해 주세요",
        datetime.datetime(2025, 1, 1)
    )

with col_end_date:
    end_date = st.date_input(
        "조회 종료일을 선택해 주세요",
        datetime.datetime.now()
    )

# 시작 날짜와 종료 날짜 비교
if start_date > end_date:
    st.warning("시작일이 종료일보다 더 늦습니다. 날짜를 자동으로 맞바꿔 반영합니다.")
    start_date, end_date = end_date, start_date  # 날짜를 바꿈

# 세 개의 종목 코드 입력 필드
#아래 '종목코드를 입력하세요' 글자 스타일 변경
st.markdown("""
    <style>
    input::placeholder {
        font-size: 13px; /* 글자 크기 */
        color: grey; /* 글자 색상 */
    }
    </style>
    """, unsafe_allow_html=True)

col_code1, col_code2, col_code3 = st.columns(3)

with col_code1:
    code1 = st.text_input('종목코드 1', value='QQQ', placeholder='종목코드를 입력하세요 - (예시)QQQ') # 기본값을 'QQQ'로 설정

with col_code2:
    code2 = st.text_input('종목코드 2', value='', placeholder='종목코드를 입력하세요 - (예시)005930')

with col_code3:
    code3 = st.text_input('종목코드 3', value='', placeholder='종목코드를 입력하세요 - (예시)AAPL')

# 세 입력 필드가 모두 입력된 후에 트래킹 코드가 작동 - 제거됨
# streamlit_analytics.track()

# 종목 코드 리스트
codes = [code1.strip(), code2.strip(), code3.strip()]

# 지수 코드 리스트 (필요에 따라 확장 가능)
# 대표적인 지수들 예시
index_codes = ['KS11', 'DJI', 'JP225', 'KQ11', 'IXIC', 'STOXX50E', 'KS50', 'GSPC', 'CSI300', 'KS100', 'S&P500', 'VIX', 'KOSPI100', 'HSI', 'KRX100', 'FTSE', 'KS200', 'DAX', 'CAC', 'GSPC'] 

# 종목 정보 가져오기
stocks_info = {}
for code in codes:
    if code:
        try:
            # 한국 종목 코드에 .KS 추가, 미국 종목은 그대로 사용, 지수는 ^를 붙여서 사용
            if code.isdigit():
                stock = yf.Ticker(f"{code}.KS")
            elif code in index_codes:  # 지수 목록에 있는 경우에는 ^ 추가
                # FinanceDataReader가 아닌 yfinance를 사용하는 경우, 지수 코드를 그대로 사용하거나 ^를 붙여야 할 수 있습니다.
                # 여기서는 yfinance를 사용하므로 ^를 붙이는 것이 일반적이지만, fdr/yf의 작동 방식에 따라 다를 수 있습니다.
                # yfinance의 지수 코드를 위해 ^를 붙이도록 수정합니다. (예: ^GSPC)
                stock = yf.Ticker(f"^{code}")
            else:
                stock = yf.Ticker(code)

            # 종목 이름 가져오기
            stock_name = stock.info.get('shortName', '이름을 찾을 수 없습니다.')
            if stock_name == '이름을 찾을 수 없습니다.' and 'longName' in stock.info:
                 stock_name = stock.info.get('longName')
                 
            stocks_info[code] = stock_name

        except Exception as e:
            # st.error(f"Error fetching info for {code}: {e}") # 디버깅용
            stocks_info[code] = '이름을 찾을 수 없습니다.'

# 종목 코드와 이름 좌우 배열로 표시
col_name1, col_name2, col_name3 = st.columns(3)

with col_name1:
    st.markdown(f"<span style='color: black;'>{code1}({stocks_info.get(code1.strip(), '')})</span>", unsafe_allow_html=True)

with col_name2:
    st.markdown(f"<span style='color: black;'>{code2}({stocks_info.get(code2.strip(), '')})</span>", unsafe_allow_html=True)

with col_name3:
    st.markdown(f"<span style='color: black;'>{code3}({stocks_info.get(code3.strip(), '')})</span>", unsafe_allow_html=True)

# '기준시점 수익률 비교' 체크박스
fixed_ratio = st.checkbox("기준시점 수익률 비교(Baseline return)", value=True) # 기본값 True로 변경

# 수평선 추가
st.markdown("---")

# 체크박스 그룹 (순서 변경)
col1, col2, col3, col4 = st.columns(4)
with col1:
    show_us_etf = st.checkbox("미국ETF", value=False)  # 미국ETF
with col2:
    show_kr_etf = st.checkbox("한국ETF", value=False)  # 한국ETF
with col3:
    show_major_stocks = st.checkbox("주요종목", value=False)  # 주요종목
with col4:
    show_major_index = st.checkbox("지수", value=False)  # 지수

# 입력된 종목 코드를 리스트로 생성
codes = [code1, code2, code3]
codes = [code.strip() for code in codes if code]  # 빈 코드 제거

# '미국ETF' 체크박스와 연결된 데이터 행렬
data_matrix_us_etf = [
    ['-3X', '-2X', '-1X', '코드', '1X', '2X', '3X'],  # 1행
    ['SPXU', 'SDS', 'SH', 'S&P500', 'SPY', 'SSO', 'UPRO'],  # 2행
    ['SQQQ', 'QID', 'PSQ', '나스닥100', 'QQQ', 'QLD', 'TQQQ'],  # 3행
    ['SDOW', 'DXD', 'DOG', '다우존스', 'DIA', 'DDM', 'UDOW'],  # 4행
    ['TZA', 'TWM', 'RWM', '러셀2000', 'IWM', 'UWM', 'TNA'],  # 5행
    ['', '', '', '한국', 'EWY', 'KORU', ''],  # 6행
    ['YANG', 'FXP', 'CHAD', '중국', 'FXI', 'CHAU', 'YINN'],  # 7행
    ['', 'EWV', '', '일본', 'EWJ', 'EZJ', 'JPNL'],  # 8행
    ['', '', '', '베트남', 'VNM', '', ''],  # 9행
    ['INDZ', '', '', '인도', 'INDA', '', 'INDL'],  # 10행
    ['RUSS', '', '', '러시아', 'RSX', '', 'RUSL'],  # 11행
    ['', 'BZQ', '', '브라질', 'EWZ', '', 'BRZU'],  # 12행
    ['DGLD', 'GLL', 'DGZ', '금', 'GLD', 'DGP', 'UGLD'],  # 13행
    ['DSLV', 'ZSL', '', '은', 'SLV', 'AGQ', 'USLV'],  # 14행
    ['DWT', 'SCO', '', '원유', 'USO', 'UCO', ''],  # 15행
    ['DGAZ', 'KOLD', '', '천연가스', 'UNG', 'BOIL', 'UGAZ'],  # 16행
    ['', '', '', '농산물', 'DBA', '', ''],  # 17행
]

# '지수' 체크박스와 연결된 데이터 행렬
data_matrix_index = [
    ['한국코드', '설명', '미국코드', '설명', '기타코드', '설명'],  # 1행
    ['KS11', 'KOSPI지수', 'DJI', '다우존스', 'JP225', '닛케이225선물'],  # 2행
    ['KQ11', 'KOSDAQ지수', 'IXIC', '나스닥', 'STOXX50E', 'EuroStoxx50'],  # 3행
    ['KS50', 'KOSPI50지수', 'GSPC', 'S&P500', 'CSI300', 'CSI300(중국)'],  # 4행
    ['KS100', 'KOSPI100', 'VIX', 'S&P500VIX', 'HSI', '항셍(홍콩)'],  # 5행
    ['KRX100', 'KRX100', '-', '-', 'FTSE', '영국FTSE'],  # 6행
    ['KS200', '코스피200', '-', '-', 'DAX', '독일DAX30'],  # 7행
]

# 시가총액 상위 11개 종목 데이터 행렬
data_matrix_top_stocks = [
    ['미국종목코드', '설명', '한국종목코드', '설명'],  # 1행
    ['AAPL', '애플', '005930', '삼성전자'],  # 2행
    ['MSFT', '마이크로소프트', '000660', 'SK하이닉스'],  # 3행
    ['AMZN', '아마존', '373220', 'LG에너지솔루션'],  # 4행
    ['NVDA', '엔비디아', '207940', '삼성바이오로직스'],  # 5행
    ['GOOGL', '알파벳A', '005380', '현대차'],  # 6행
    ['META', '메타', '068270', '셀트리온'],  # 7행
    ['TSLA', '테슬라', '000270', '기아'],  # 8행
    ['BRK.B', '버크셔헤서웨이', '196170', '알테오젠'],  # 9행
    ['UNH', '유나이티드헬스', '247540', '에코프로비엠'],  # 10행
    ['JNJ', '존슨앤존슨', '086520', '에코프로'],  # 11행
]

# 한국ETF 체크박스와 연결된 데이터 행렬 (4열)
data_matrix_kr_etf = [
    ['한국종목코드', '설명', '한국종목코드', '설명'],  # 열 제목
    ['069500', 'KODEX 200', '122630', 'KODEX 레버리지'],
    ['229200', 'KODEX 코스닥150', '233740', 'KODEX 코스닥150레버리지'],
    ['114800', 'KODEX 인버스', '252670', 'KODEX 200선물인버스2X'],
    ['251340', 'KODEX 코스닥150선물인버스', '442580', 'PLUS 글로벌HBM반도체'],
    ['243890', 'TIGER 200에너지화학레버리지', '412570', 'TIGER 2차전지TOP10레버리지'],
    ['463640', 'KODEX 미국S&P500유틸리티', '379800', 'KODEX 미국S&P500TR'],
    ['379810', 'KODEX 미국나스닥100TR', '449190', 'KODEX 미국나스닥100(H)'],
    ['409820', 'KODEX 미국나스닥100레버리지(합성 H)', '438100', 'ACE 미국나스닥100채권혼합액티브'],
    ['447660', 'PLUS 애플채권혼합', '448540', 'ACE 엔비디아채권혼합블룸버그'],
    ['236350', 'TIGER 인도니프티50레버리지(합성)', '132030', 'KODEX 골드선물(H)'],
    ['144600', 'KODEX 은선물(H)', '530063', '삼성 레버리지 구리 선물 ETN(H)'],
    ['530031', '삼성 레버리지 WTI원유 선물 ETN', '530036', '삼성 인버스 2X WTI원유 선물 ETN'],
    ['438320', 'TIGER 차이나항셍테크레버리지(합성 H)', '371460', 'TIGER 차이나전기차SOLACTIVE'],
]

# '주요종목' 체크박스를 선택할 때 표 출력
if show_major_index or show_major_stocks or show_us_etf or show_kr_etf:
    # 미국ETF 표 출력
    if show_us_etf:
        st.markdown("<h4 style='font-size: 16px; text-align: left; margin-top: 20px;'>미국 ETF 주요 코드</h4>", unsafe_allow_html=True)
        html = '''
        <style>
        table {
            border-collapse: collapse;  
            width: 100%;  
            font-size: 10px;  /* 글자 크기를 10px로 설정 */
        }
        td {
            border: 1px solid black;  
            padding: 8px;  
            text-align: center;
            /* color: #333333; <-- 일반 셀의 텍스트 색상 고정 제거 */
        }
        .highlight {
            background-color: lightgray;
            color: #333333; /* 다크 모드에서도 대비가 확보되도록 텍스트 색상을 어두운 회색으로 고정 */
        }
        </style>
        <table>
        '''
        
        for i, row in enumerate(data_matrix_us_etf):
            html += '<tr>'
            for j, cell in enumerate(row):
                if i == 0 or j == 3:  # 첫 번째 행과 코드 열 강조
                    html += f'<td class="highlight">{cell}</td>'
                else:
                    html += f'<td>{cell}</td>'
            html += '</tr>'
        html += '</table>'
        st.markdown(html, unsafe_allow_html=True)

    # 한국ETF 표 출력
    if show_kr_etf:
        st.markdown("<h4 style='font-size: 16px; text-align: left; margin-top: 20px;'>한국 ETF 주요 코드</h4>", unsafe_allow_html=True)
        html_kr_etf = '''
        <style>
        table {
            border-collapse: collapse;  
            width: 100%;  
            font-size: 10px;  /* 글자 크기를 10px로 설정 */
        }
        td {
            border: 1px solid black;  
            padding: 8px;  
            text-align: center;
            /* color: #333333; <-- 일반 셀의 텍스트 색상 고정 제거 */
        }
        .highlight {
            background-color: lightgray;
            color: #333333; /* 다크 모드에서도 대비가 확보되도록 텍스트 색상을 어두운 회색으로 고정 */
        }
        </style>
        <table>
        '''
        
        for i, row in enumerate(data_matrix_kr_etf):
            html_kr_etf += '<tr>'
            for j, cell in enumerate(row):
                if i == 0 or j == 1 or j == 3:  # 첫 번째 행과 코드 열 강조
                    html_kr_etf += f'<td class="highlight">{cell}</td>'
                else:
                    html_kr_etf += f'<td>{cell}</td>'
            html_kr_etf += '</tr>'
        html_kr_etf += '</table>'
        st.markdown(html_kr_etf, unsafe_allow_html=True)
    
    # 주요종목 표 출력
    if show_major_stocks:
        st.markdown("<h4 style='font-size: 16px; text-align: left; margin-top: 20px;'>주요 시가총액 상위 종목 코드</h4>", unsafe_allow_html=True)
        html_major_stocks = '''
        <style>
        table {
            border-collapse: collapse;  
            width: 100%;  
            font-size: 10px;  /* 글자 크기를 10px로 설정 */
        }
        td {
            border: 1px solid black;  
            padding: 8px;  
            text-align: center;
            /* color: #333333; <-- 일반 셀의 텍스트 색상 고정 제거 */
        }
        .highlight {
            background-color: lightgray;
            color: #333333; /* 다크 모드에서도 대비가 확보되도록 텍스트 색상을 어두운 회색으로 고정 */
        }
        </style>
        <table>
        '''
        
        for i, row in enumerate(data_matrix_top_stocks):
            html_major_stocks += '<tr>'
            for j, cell in enumerate(row):
                if i == 0 or j == 1 or j == 3:  # 첫 번째 행과 코드 열 강조
                    html_major_stocks += f'<td class="highlight">{cell}</td>'
                else:
                    html_major_stocks += f'<td>{cell}</td>'
            html_major_stocks += '</tr>'
        html_major_stocks += '</table>'
        st.markdown(html_major_stocks, unsafe_allow_html=True)

    # 지수 표 출력
    if show_major_index:
        st.markdown("<h4 style='font-size: 16px; text-align: left; margin-top: 20px;'>주요 지수 코드</h4>", unsafe_allow_html=True)
        html_index = '''
        <style>
        table {
            border-collapse: collapse;  
            width: 100%;  
            font-size: 10px;  /* 글자 크기를 10px로 설정 */
        }
        td {
            border: 1px solid black;  
            padding: 8px;  
            text-align: center;
            /* color: #333333; <-- 일반 셀의 텍스트 색상 고정 제거 */
        }
        .highlight {
            background-color: lightgray;
            color: #333333; /* 다크 모드에서도 대비가 확보되도록 텍스트 색상을 어두운 회색으로 고정 */
        }
        </style>
        <table>
        '''
        
        for i, row in enumerate(data_matrix_index):
            html_index += '<tr>'
            for j, cell in enumerate(row):
                if i == 0 or j == 1 or j == 3 or j == 5:  # 첫 번째 행과 코드 열 강조
                    html_index += f'<td class="highlight">{cell}</td>'
                else:
                    html_index += f'<td>{cell}</td>'
            html_index += '</tr>'
        html_index += '</table>'
        st.markdown(html_index, unsafe_allow_html=True)

# 데이터 로딩 부분에서 오류 처리
if codes and start_date and end_date:  # 'date'를 'start_date'와 'end_date'로 수정
    dataframes = []
    
    for code in codes:
        try:
            df = fdr.DataReader(code, start_date, end_date)  # 'date'를 'start_date', 'end_date'로 수정
            close_prices = df['Close']
            if fixed_ratio:
                start_price = close_prices.iloc[0]
                data = ((close_prices - start_price) / start_price) * 100
                dataframes.append(data.rename(code))
            else:
                dataframes.append(close_prices.rename(code))
        except Exception:  # Exception을 처리하되, 오류 메시지를 표시하지 않음
            st.warning(f"**{code}**의 데이터를 불러오는 데 문제가 발생했습니다. 종목 코드와 날짜를 확인해 주세요.")

    # 데이터프레임 리스트가 있을 경우
    if dataframes and not all(df.empty for df in dataframes):
        combined_data = pd.concat(dataframes, axis=1)
        tab1, tab2 = st.tabs(['차트', '데이터'])
    
        with tab1:
            st.line_chart(combined_data, use_container_width=True)
            if fixed_ratio:
                st.write("Y축은 비율로 표시되며, 기준시점 0% 에서 시작합니다.")
    
        with tab2:
            try:
                # 모든 종목의 전체 데이터를 병합합니다.
                all_data = pd.concat(
                    [fdr.DataReader(code, start_date, end_date).rename(columns=lambda x: f"{x} ({code})") for code in codes], 
                    axis=1
                )
                st.dataframe(all_data)
            except Exception:
                 st.warning("데이터 탭에 모든 상세 데이터를 표시하는 데 문제가 발생했습니다.")
            
            # 컬럼 설명을 표 형식으로 표시
            column_description = {
                '컬럼명': ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'],
                '설명': ['시가', '고가', '저가', '종가', '수정 종가', '거래량']
            }
            description_df = pd.DataFrame(column_description)
            st.table(description_df)

# 조회 시작일 가상 투자금액의 수익률 및 수익금액 계산(진행중)

# 수익금액 비교 입력
st.markdown("<h2 style='font-size: 20px; margin-top: 30px;'>1천만원을 투자했다면,</h2>", unsafe_allow_html=True)

# 고정된 투자금액 (1000만원)
initial_investment = 10000000  

# 각 종목의 수익률 및 수익금액 계산
if codes and start_date and end_date:
    profit_info = []

    for code in codes:
        try:
            df = fdr.DataReader(code, start_date, end_date)
            close_prices = df['Close']
            start_price = close_prices.iloc[0]  # 시작 가격
            end_price = close_prices.iloc[-1]    # 종료 가격

            # 수익률 계산
            return_percentage = ((end_price - start_price) / start_price) * 100
            # 수익금액 계산
            profit_amount = (return_percentage / 100) * initial_investment
            total_amount = initial_investment + profit_amount

            # 종목 이름 가져오기
            stock_name = stocks_info.get(code.strip(), '이름을 찾을 수 없습니다.')

            # 수익률 색상 결정
            if return_percentage >= 0:
                color = 'red'  # 양수일 경우 붉은색
            else:
                color = 'blue'   # 음수일 경우 파란색

            # 투자결과 출력
            profit_info.append(f"**{stock_name}** ({code})에 투자한 결과 **{total_amount:,.0f} 원**이 되었습니다. (<span style='color: {color};'>수익률: **{return_percentage:.2f}%**</span>)")
            
        except Exception:
            profit_info.append(f"**{code}**의 데이터를 불러오는 데 문제가 발생했습니다.")

    # 수익률 및 수익금액 결과 출력
    for info in profit_info:
        st.markdown(info, unsafe_allow_html=True)  # HTML을 안전하게 허용하여 출력
