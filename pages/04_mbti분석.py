
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

# ==========================================
# 1. 기본 설정
# ==========================================
st.set_page_config(
    page_title="세계 MBTI 국가별 분석",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 세계 MBTI 국가별 분석")
st.write(
    "국가를 선택하고 MBTI 유형별 비율을 확인해 보세요!"
)

MBTI_TYPES = [
    "INFJ", "ISFJ", "INTP", "ISFP",
    "ENTP", "INFP", "ENTJ", "ISTP",
    "INTJ", "ESFP", "ESTJ", "ENFP",
    "ESTP", "ISTJ", "ENFJ", "ESFJ"
]

# ==========================================
# 2. CSV 파일 경로 설정
# ==========================================

# 현재 파일 위치: aiproject/pages/03_국가별MBTI.py
# 부모 폴더 위치: aiproject/
BASE_DIR = Path(__file__).resolve().parent.parent

CSV_PATH = BASE_DIR / "countriesMBTI_16types.csv"


# ==========================================
# 3. CSV 데이터 불러오기
# ==========================================
@st.cache_data
def load_data(file_path):

    df = pd.read_csv(
        file_path,
        encoding="utf-8-sig"
    )

    required_columns = ["Country"] + MBTI_TYPES

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "CSV 파일에 필요한 열이 없습니다: "
            + ", ".join(missing_columns)
        )

    df = df[required_columns].copy()

    df["Country"] = (
        df["Country"]
        .astype(str)
        .str.strip()
    )

    for mbti in MBTI_TYPES:
        df[mbti] = pd.to_numeric(
            df[mbti],
            errors="coerce"
        )

    df = df.dropna(
        subset=["Country"] + MBTI_TYPES
    )

    df = df[df["Country"] != ""]

    if df.empty:
        raise ValueError(
            "분석할 데이터가 없습니다."
        )

    return df


# ==========================================
# 4. CSV 파일 존재 여부 확인
# ==========================================
if not CSV_PATH.exists():

    st.error(
        "CSV 파일을 찾을 수 없습니다."
    )

    st.write(
        f"찾으려는 경로: {CSV_PATH}"
    )

    st.info(
        "countriesMBTI_16types.csv 파일이 "
        "프로젝트 최상위 폴더에 있는지 확인하세요."
    )

    st.stop()


try:

    df = load_data(str(CSV_PATH))

except Exception as error:

    st.error(
        f"데이터를 불러오지 못했습니다: {error}"
    )

    st.stop()


# ==========================================
# 5. 국가 선택
# ==========================================
st.sidebar.header("🌎 국가 선택")

countries = sorted(
    df["Country"].unique().tolist()
)

default_index = (
    countries.index("South Korea")
    if "South Korea" in countries
    else 0
)

selected_country = st.sidebar.selectbox(
    "분석할 국가를 선택하세요.",
    countries,
    index=default_index
)


# ==========================================
# 6. 선택 국가 데이터 추출
# ==========================================
country_row = df.loc[
    df["Country"] == selected_country
].iloc[0]


# ==========================================
# 7. MBTI 비율 계산 및 정렬
# ==========================================
chart_df = pd.DataFrame({

    "MBTI": MBTI_TYPES,

    "비율": [
        float(country_row[mbti]) * 100
        for mbti in MBTI_TYPES
    ]

})

chart_df = chart_df.sort_values(
    by="비율",
    ascending=False
).reset_index(drop=True)


top_mbti = chart_df.loc[0, "MBTI"]

top_percentage = chart_df.loc[0, "비율"]


# ==========================================
# 8. 주요 정보 표시
# ==========================================
col1, col2, col3 = st.columns(3)

col1.metric(
    "선택 국가",
    selected_country
)

col2.metric(
    "가장 높은 MBTI",
    top_mbti
)

col3.metric(
    "1위 비율",
    f"{top_percentage:.2f}%"
)

st.divider()


# ==========================================
# 9. 그래프 색상 설정
# ==========================================
blue_gradient = [

    "#1769D2",
    "#2878D8",
    "#3987DE",
    "#4A96E4",
    "#5BA5E9",
    "#6CB4ED",
    "#7DC2F1",
    "#8ED0F4",
    "#9FD9F6",
    "#AFE0F8",
    "#BDE6FA",
    "#C9EBFB",
    "#D3EFFC",
    "#DDF3FD",
    "#E7F7FE"

]

# 1위는 빨간색
# 나머지는 파란색 그라데이션
bar_colors = [
    "#E53935"
] + blue_gradient


# ==========================================
# 10. Plotly 막대그래프
# ==========================================
fig = go.Figure()

fig.add_trace(

    go.Bar(

        x=chart_df["MBTI"],

        y=chart_df["비율"],

        marker=dict(
            color=bar_colors,
            line=dict(width=0)
        ),

        text=[
            f"{value:.2f}%"
            for value in chart_df["비율"]
        ],

        textposition="outside",

        cliponaxis=False,

        hovertemplate=(
            "<b>%{x}</b><br>"
            "비율: %{y:.2f}%"
            "<extra></extra>"
        )

    )

)


# ==========================================
# 11. 그래프 디자인
# ==========================================
fig.update_layout(

    title=dict(
        text=f"{selected_country}의 MBTI 유형별 비율",
        x=0.02,
        xanchor="left"
    ),

    xaxis_title="MBTI 유형",

    yaxis_title="비율(%)",

    yaxis=dict(
        rangemode="tozero",
        ticksuffix="%",
        showgrid=True
    ),

    xaxis=dict(
        categoryorder="array",
        categoryarray=chart_df["MBTI"].tolist()
    ),

    showlegend=False,

    height=570,

    margin=dict(
        l=25,
        r=25,
        t=75,
        b=45
    ),

    hoverlabel=dict(
        namelength=-1
    )

)


# ==========================================
# 12. 그래프 출력
# ==========================================
st.subheader("📊 국가별 MBTI 비율")

st.caption(
    "1위는 빨간색, 나머지는 순위가 낮아질수록 "
    "파란색이 점점 연해집니다."
)

st.plotly_chart(
    fig,
    use_container_width=True,
    config={
        "displayModeBar": True
    }
)


# ==========================================
# 13. MBTI 전체 순위표
# ==========================================
st.divider()

st.subheader("🏆 MBTI 비율 순위")

display_df = chart_df.copy()

display_df.insert(
    0,
    "순위",
    range(1, len(display_df) + 1)
)

display_df["비율"] = display_df["비율"].map(
    lambda value: f"{value:.2f}%"
)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)


# ==========================================
# 14. 하단 안내
# ==========================================
st.divider()

st.caption(
    "※ 본 자료는 제공된 CSV의 국가별 MBTI 분포를 "
    "시각화한 것입니다."
)

st.caption(
    "※ 개인의 성격을 판단하거나 예측하는 자료는 아닙니다."
)
