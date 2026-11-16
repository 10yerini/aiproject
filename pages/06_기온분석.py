
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

# --------------------------------
# 1. 페이지 설정
# --------------------------------

st.set_page_config(
    page_title="월별 기온분석",
    page_icon="🌡️",
    layout="wide"
)

# --------------------------------
# 2. 제목
# --------------------------------

st.title("🌡️ 월별 기온분석")
st.write("원하는 월을 선택하여 연도별 최고기온과 최저기온을 비교해 보세요!")

st.divider()

# --------------------------------
# 3. CSV 파일 불러오기
# --------------------------------

# pages 폴더의 상위 폴더에서 CSV 찾기
BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "temperature.csv"

try:
    try:
        df = pd.read_csv(CSV_PATH, encoding="cp949")
    except UnicodeDecodeError:
        df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")

except FileNotFoundError:
    st.error("temperature.csv 파일을 찾을 수 없습니다.")
    st.stop()

# 컬럼 이름 공백 제거
df.columns = df.columns.str.strip()

# --------------------------------
# 4. 날짜 데이터 처리
# --------------------------------

required_columns = [
    "일시",
    "최고기온(℃)",
    "최저기온(℃)"
]

if not all(col in df.columns for col in required_columns):
    st.error("CSV 파일의 컬럼 이름을 확인해 주세요.")
    st.write("현재 컬럼:", df.columns.tolist())
    st.stop()

# Oct-07 형식의 날짜 변환
df["날짜"] = pd.to_datetime(
    df["일시"].astype(str).str.strip(),
    format="%b-%y",
    errors="coerce"
)

# 연도와 월 추출
df["연도"] = df["날짜"].dt.year
df["월"] = df["날짜"].dt.month

# 기온 데이터 숫자로 변환
df["최고기온(℃)"] = pd.to_numeric(
    df["최고기온(℃)"],
    errors="coerce"
)

df["최저기온(℃)"] = pd.to_numeric(
    df["최저기온(℃)"],
    errors="coerce"
)

# 날짜가 정상적으로 변환된 데이터만 사용
df = df.dropna(subset=["날짜"]).copy()

if df.empty:
    st.error("날짜 데이터를 읽을 수 없습니다.")
    st.stop()

# --------------------------------
# 5. 월 선택
# --------------------------------

month_names = [
    "1월", "2월", "3월", "4월",
    "5월", "6월", "7월", "8월",
    "9월", "10월", "11월", "12월"
]

selected_month = st.selectbox(
    "📅 분석할 월을 선택하세요.",
    range(1, 13),
    format_func=lambda x: month_names[x - 1]
)

# --------------------------------
# 6. 선택한 월 데이터 추출
# --------------------------------

month_df = df[df["월"] == selected_month].copy()

month_df = month_df.sort_values("연도")

if month_df.empty:
    st.warning("선택한 월의 데이터가 없습니다.")
    st.stop()

# --------------------------------
# 7. 꺾은선 그래프
# --------------------------------

fig = go.Figure()

# 최고기온: 핑크색
high_df = month_df.dropna(
    subset=["최고기온(℃)"]
)

fig.add_trace(
    go.Scatter(
        x=high_df["연도"],
        y=high_df["최고기온(℃)"],
        mode="lines+markers",
        name="최고기온",
        line=dict(
            color="#FF69B4",
            width=3
        ),
        marker=dict(size=8),
        connectgaps=False
    )
)

# 최저기온: 하늘색
low_df = month_df.dropna(
    subset=["최저기온(℃)"]
)

fig.add_trace(
    go.Scatter(
        x=low_df["연도"],
        y=low_df["최저기온(℃)"],
        mode="lines+markers",
        name="최저기온",
        line=dict(
            color="#87CEEB",
            width=3
        ),
        marker=dict(size=8),
        connectgaps=False
    )
)

# --------------------------------
# 8. 그래프 디자인
# --------------------------------

fig.update_layout(
    title=dict(
        text="월별 기온분석",
        x=0.5,
        xanchor="center",
        font=dict(size=25)
    ),

    xaxis=dict(
        title="연도",
        dtick=1,
        tickformat="d",
        showgrid=True
    ),

    yaxis=dict(
        title="온도(℃)",
        showgrid=True
    ),

    legend=dict(
        title="기온 구분",
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5
    ),

    hovermode="x unified",
    template="plotly_white",
    font=dict(
        family="Malgun Gothic, Arial, sans-serif",
        size=13
    ),

    height=550
)

# --------------------------------
# 9. 그래프 출력
# --------------------------------

st.subheader(f"📊 {selected_month}월 연도별 기온 변화")

st.plotly_chart(
    fig,
    use_container_width=True
)

# --------------------------------
# 10. 데이터 표 출력
# --------------------------------

with st.expander("📋 기온 데이터 자세히 보기"):

    result = month_df[
        ["연도", "최고기온(℃)", "최저기온(℃)"]
    ].copy()

    result["연도"] = result["연도"].astype(int)

    st.dataframe(
        result,
        use_container_width=True,
        hide_index=True
    )

st.divider()

st.caption("🌡️ 기온 데이터 분석 프로그램")
