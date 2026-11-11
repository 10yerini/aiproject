
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# --------------------------------------------------
# 1. 페이지 설정
# --------------------------------------------------
st.set_page_config(
    page_title="서울시 연령별 인구 분석",
    page_icon="📊",
    layout="wide"
)

st.title("📊 서울시 연령별 인구 분석")
st.caption("행정구역, 연도, 성별을 선택해 연령대별 인구 분포를 살펴보세요.")

# --------------------------------------------------
# 2. CSV 파일 경로
# PY 파일은 pages 폴더 안에 있음
# CSV 파일은 프로젝트 최상위 폴더에 있음
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "population (1).csv"

# --------------------------------------------------
# 3. CSV 파일 읽기 (한글 인코딩 대응)
# --------------------------------------------------
@st.cache_data
def load_population_data(file_path):
    encodings = ["cp949", "utf-8-sig", "utf-8"]

    for encoding in encodings:
        try:
            return pd.read_csv(
                file_path,
                encoding=encoding
            )
        except UnicodeDecodeError:
            continue

    raise ValueError("CSV 파일의 인코딩을 확인해주세요.")


# 파일 존재 여부 확인
if not CSV_PATH.exists():

    st.error("population (1).csv 파일을 찾을 수 없습니다.")

    st.info(
        "CSV 파일이 프로젝트 최상위 폴더에 "
        "있는지 확인해주세요."
    )

    st.code("""
aiproject/
│
├── population (1).csv
├── requirements.txt
│
└── pages/
    └── 04_인구분석.py
""")

    st.stop()


# CSV 읽기
try:
    df = load_population_data(str(CSV_PATH))

except Exception as e:
    st.error(f"데이터 파일 읽기 오류: {e}")
    st.stop()


# --------------------------------------------------
# 4. 데이터 전처리
# --------------------------------------------------

if "행정구역" not in df.columns:
    st.error("'행정구역' 컬럼을 찾을 수 없습니다.")
    st.stop()


# 행정구역명에서 코드 제거
# 예: 서울특별시 종로구 (1111000000)
df["행정구역명"] = (
    df["행정구역"]
    .astype(str)
    .str.replace(r"\s*\(\d+\)$", "", regex=True)
    .str.strip()
)


# 연령대 목록
age_groups = [
    "0~9세",
    "10~19세",
    "20~29세",
    "30~39세",
    "40~49세",
    "50~59세",
    "60~69세",
    "70~79세",
    "80~89세",
    "90~99세",
    "100세 이상"
]


# 숫자 데이터 변환
for col in df.columns:

    if col not in ["행정구역", "행정구역명"]:

        df[col] = pd.to_numeric(
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.strip(),
            errors="coerce"
        )


# --------------------------------------------------
# 5. 데이터에 있는 연도 자동 추출
# --------------------------------------------------

available_years = sorted(
    {
        col.split("년")[0]
        for col in df.columns
        if "년_거주자_" in col
        and col.split("년")[0].isdigit()
    }
)

if not available_years:
    st.error("연도별 인구 데이터를 찾을 수 없습니다.")
    st.stop()


# --------------------------------------------------
# 6. 행정구 / 연도 / 성별 선택
# --------------------------------------------------

st.subheader("🔎 분석 조건 선택")

col1, col2, col3 = st.columns(3)

with col1:

    district_options = (
        df["행정구역명"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_district = st.selectbox(
        "🏙️ 행정구 선택",
        district_options
    )


with col2:

    selected_year = st.selectbox(
        "📅 연도 선택",
        available_years,
        index=len(available_years) - 1
    )


with col3:

    selected_gender = st.selectbox(
        "👥 성별 선택",
        ["전체", "남성", "여성"]
    )


# --------------------------------------------------
# 7. 성별에 따른 컬럼 선택
# --------------------------------------------------

if selected_gender == "전체":

    column_prefix = (
        f"{selected_year}년_거주자_"
    )

elif selected_gender == "남성":

    column_prefix = (
        f"{selected_year}년_남_거주자_"
    )

else:

    column_prefix = (
        f"{selected_year}년_여_거주자_"
    )


# 선택한 행정구 데이터
selected_data = df[
    df["행정구역명"] == selected_district
]

if selected_data.empty:
    st.error("선택한 행정구역의 데이터가 없습니다.")
    st.stop()

selected_row = selected_data.iloc[0]


# --------------------------------------------------
# 8. 연령대별 인구 데이터 생성
# --------------------------------------------------

chart_rows = []

for age in age_groups:

    column_name = column_prefix + age

    if column_name in df.columns:

        value = selected_row[column_name]

        chart_rows.append({
            "나이": age,
            "인구수": (
                0 if pd.isna(value)
                else int(value)
            )
        })


chart_df = pd.DataFrame(chart_rows)

if chart_df.empty:
    st.error("연령별 인구 데이터가 없습니다.")
    st.stop()


# --------------------------------------------------
# 9. 요약 정보
# --------------------------------------------------

total_population = int(
    chart_df["인구수"].sum()
)

max_age_row = chart_df.loc[
    chart_df["인구수"].idxmax()
]

st.divider()

st.subheader("📌 인구 분석 요약")

metric1, metric2, metric3 = st.columns(3)

metric1.metric(
    "선택한 행정구",
    selected_district
)

metric2.metric(
    "연령별 인구 합계",
    f"{total_population:,}명"
)

metric3.metric(
    "인구가 가장 많은 연령대",
    max_age_row["나이"],
    f"{int(max_age_row['인구수']):,}명"
)


# --------------------------------------------------
# 10. 초록색 꺾은선 그래프
# --------------------------------------------------

st.divider()

st.subheader(
    f"📈 {selected_district} "
    f"{selected_year}년 "
    f"{selected_gender} 연령별 인구"
)


fig = px.line(
    chart_df,
    x="나이",
    y="인구수",
    markers=True,
    color_discrete_sequence=["#16A34A"],
    labels={
        "나이": "나이",
        "인구수": "인구수 (명)"
    }
)


# 선과 점 설정
fig.update_traces(
    line=dict(
        color="#16A34A",
        width=3
    ),
    marker=dict(
        color="#16A34A",
        size=9
    ),
    hovertemplate=(
        "나이: %{x}<br>"
        "인구수: %{y:,.0f}명"
        "<extra></extra>"
    )
)


# 세로축 1,000명 단위 설정
max_population = int(
    chart_df["인구수"].max()
)

y_axis_max = max(
    1000,
    ((max_population // 1000) + 1) * 1000
)


fig.update_layout(

    template="plotly_white",

    height=560,

    showlegend=False,

    hovermode="x unified",

    margin=dict(
        l=20,
        r=20,
        t=30,
        b=20
    ),

    # 가로축
    xaxis=dict(
        title="나이",
        categoryorder="array",
        categoryarray=age_groups,
        showgrid=False
    ),

    # 세로축
    yaxis=dict(
        title="인구수 (명)",

        range=[
            0,
            y_axis_max
        ],

        tickmode="linear",

        tick0=0,

        dtick=1000,

        tickformat=",d",

        showgrid=True,

        gridcolor="#DCEEDC",

        gridwidth=1,

        zeroline=True,

        zerolinecolor="#A3C9A3"
    )
)


# 그래프 출력
st.plotly_chart(
    fig,
    use_container_width=True
)


# --------------------------------------------------
# 11. 연령별 인구 데이터 표
# --------------------------------------------------

st.divider()

with st.expander("📋 연령대별 인구 데이터 보기"):

    display_df = chart_df.copy()

    display_df["인구수"] = (
        display_df["인구수"]
        .map(lambda x: f"{x:,}명")
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )


    # CSV 다운로드
    csv_data = chart_df.to_csv(
        index=False
    ).encode("utf-8-sig")


    st.download_button(
        label="⬇️ 분석 결과 CSV 다운로드",

        data=csv_data,

        file_name=(
            f"{selected_district}_"
            f"{selected_year}_"
            f"{selected_gender}_연령별인구.csv"
        ),

        mime="text/csv"
    )


# --------------------------------------------------
# 12. 하단 안내
# --------------------------------------------------

st.divider()

st.caption(
    "📊 데이터 출처: population (1).csv"
)

st.caption(
    "한글 인코딩: CP949 / UTF-8 자동 대응"
)
