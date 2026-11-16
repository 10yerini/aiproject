
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

# =====================================
# 1. 페이지 설정
# =====================================

st.set_page_config(
    page_title="월별 기온분석",
    page_icon="🌡️",
    layout="wide"
)

# =====================================
# 2. 제목
# =====================================

st.title("🌡️ 월별 기온분석")

st.write(
    "원하는 월을 선택하면 2007년부터 2026년까지 "
    "연도별 최고기온과 최저기온을 비교할 수 있습니다."
)

st.divider()

# =====================================
# 3. CSV 파일 불러오기
# =====================================

# pages 폴더의 상위 폴더 경로
BASE_DIR = Path(__file__).resolve().parent.parent

CSV_PATH = BASE_DIR / "temperature.csv"

try:
    try:
        df = pd.read_csv(
            CSV_PATH,
            encoding="cp949"
        )

    except UnicodeDecodeError:
        df = pd.read_csv(
            CSV_PATH,
            encoding="utf-8-sig"
        )

except FileNotFoundError:
    st.error(
        "temperature.csv 파일을 찾을 수 없습니다. "
        "pages 폴더의 상위 폴더에 CSV 파일을 넣어 주세요."
    )
    st.stop()

# =====================================
# 4. 컬럼 이름 정리
# =====================================

df.columns = df.columns.str.strip()

required_columns = [
    "일시",
    "최고기온(℃)",
    "최저기온(℃)"
]

for col in required_columns:
    if col not in df.columns:
        st.error(f"'{col}' 컬럼을 찾을 수 없습니다.")
        st.write("현재 CSV 컬럼:", df.columns.tolist())
        st.stop()

# =====================================
# 5. 날짜 데이터 처리
# =====================================

df["날짜"] = pd.to_datetime(
    df["일시"].astype(str).str.strip(),
    format="%b-%y",
    errors="coerce"
)

# 연도와 월 추출
df["연도"] = df["날짜"].dt.year
df["월"] = df["날짜"].dt.month

# 기온을 숫자로 변환
df["최고기온(℃)"] = pd.to_numeric(
    df["최고기온(℃)"],
    errors="coerce"
)

df["최저기온(℃)"] = pd.to_numeric(
    df["최저기온(℃)"],
    errors="coerce"
)

# 날짜가 없는 행 제거
df = df.dropna(
    subset=["날짜"]
).copy()

if df.empty:
    st.error("날짜 데이터를 확인할 수 없습니다.")
    st.stop()

# =====================================
# 6. 월 선택
# =====================================

month_names = [
    "1월", "2월", "3월", "4월",
    "5월", "6월", "7월", "8월",
    "9월", "10월", "11월", "12월"
]

selected_month = st.selectbox(
    "📅 분석할 월을 선택하세요.",
    options=list(range(1, 13)),
    format_func=lambda x: month_names[x - 1]
)

# =====================================
# 7. 선택한 월의 데이터 추출
# =====================================

month_df = df[
    (df["월"] == selected_month) &
    (df["연도"] >= 2007) &
    (df["연도"] <= 2026)
].copy()

# 연도순 정렬
month_df = month_df.sort_values("연도")

if month_df.empty:
    st.warning("선택한 월의 데이터가 없습니다.")
    st.stop()

# =====================================
# 8. 그래프 생성
# =====================================

fig = go.Figure()

# 최고기온 데이터
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
        marker=dict(
            size=8,
            color="#FF69B4"
        )
    )
)

# 최저기온 데이터
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
        marker=dict(
            size=8,
            color="#87CEEB"
        )
    )
)

# =====================================
# 9. 그래프 디자인
# =====================================

fig.update_layout(

    # 그래프 제목
    title=dict(
        text="월별 기온분석",
        x=0.5,
        xanchor="center",
        font=dict(size=25)
    ),

    # 가로축
    xaxis=dict(
        title="연도",
        range=[2007, 2026],
        tickmode="linear",
        tick0=2007,
        dtick=1,
        showgrid=True
    ),

    # 세로축
    yaxis=dict(
        title="온도(℃)",
        showgrid=True
    ),

    # 범례
    legend=dict(
        title="기온 구분",
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5
    ),

    # 마우스를 올리면 데이터 표시
    hovermode="x unified",

    # 그래프 배경
    template="plotly_white",

    # 글꼴
    font=dict(
        family="Malgun Gothic, Arial, sans-serif",
        size=13
    ),

    height=550
)

# =====================================
# 10. 그래프 출력
# =====================================

st.subheader(
    f"📊 {selected_month}월 연도별 기온 변화"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================
# 11. 기온 데이터 표
# =====================================

st.subheader("📋 연도별 기온 데이터")

result = month_df[
    ["연도", "최고기온(℃)", "최저기온(℃)"]
].copy()

result["연도"] = result["연도"].astype(int)

st.dataframe(
    result,
    use_container_width=True,
    hide_index=True
)

# =====================================
# 12. 기온 통계
# =====================================

st.subheader("📈 선택한 월의 기온 통계")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "평균 최고기온",
        f"{month_df['최고기온(℃)'].mean():.1f}℃"
    )

with col2:
    st.metric(
        "평균 최저기온",
        f"{month_df['최저기온(℃)'].mean():.1f}℃"
    )

# =====================================
# 13. 마무리
# =====================================

st.divider()

st.caption(
    "🌡️ 월별 기온분석 프로그램 | 2007년~2026년"
)
