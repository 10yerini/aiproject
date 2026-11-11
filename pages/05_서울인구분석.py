
from pathlib import Path
import re

import pandas as pd
import plotly.express as px
import streamlit as st


# ==========================================
# 1. 페이지 설정
# ==========================================
st.set_page_config(
    page_title="서울시 연령별 인구 분석",
    page_icon="📊",
    layout="wide"
)

st.title("📊 서울시 연령별 인구 분석")
st.caption("행정구역, 연도, 성별에 따른 연령별 인구 분석")


# ==========================================
# 2. CSV 파일 경로
# ==========================================
# PY 파일: pages 폴더
# CSV 파일: 프로젝트 최상위 폴더

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "population (1).csv"


# ==========================================
# 3. CSV 파일 읽기
# ==========================================
@st.cache_data
def load_data(file_path):

    for encoding in ["cp949", "utf-8-sig", "utf-8"]:

        try:
            data = pd.read_csv(
                file_path,
                encoding=encoding
            )

            data.columns = [
                str(col).replace("\ufeff", "").strip()
                for col in data.columns
            ]

            return data

        except UnicodeDecodeError:
            continue

    raise ValueError("CSV 인코딩을 확인해주세요.")


if not CSV_PATH.exists():

    st.error("population (1).csv 파일을 찾을 수 없습니다.")

    st.write("현재 찾는 경로:")
    st.code(str(CSV_PATH))

    st.stop()


try:
    df = load_data(CSV_PATH)

except Exception as e:

    st.error(f"CSV 파일 읽기 오류: {e}")
    st.stop()


# ==========================================
# 4. 컬럼명 정리
# ==========================================
if "행정구역" not in df.columns:

    st.error("'행정구역' 컬럼이 없습니다.")
    st.write(df.columns.tolist())
    st.stop()


def normalize_column(name):

    name = str(name).replace("\ufeff", "").strip()

    name = re.sub(
        r"\s*_\s*",
        "_",
        name
    )

    return name


# 정규화된 컬럼명과 실제 컬럼명 연결
column_lookup = {
    normalize_column(col): col
    for col in df.columns
}


# ==========================================
# 5. 연도 자동 추출
# ==========================================
year_set = set()

for column in df.columns:

    normalized = normalize_column(column)

    match = re.match(
        r"^(\d{4})\s*년",
        normalized
    )

    if match:
        year_set.add(match.group(1))


available_years = sorted(year_set)


if not available_years:

    st.error("연도별 데이터를 찾을 수 없습니다.")

    st.write("CSV 파일의 실제 컬럼명:")

    st.code(
        "\n".join(
            str(col) for col in df.columns[:40]
        )
    )

    st.stop()


# ==========================================
# 6. 행정구역 데이터 정리
# ==========================================
df["행정구역명"] = (
    df["행정구역"]
    .astype(str)
    .str.replace(
        r"\s*\(\d+\)$",
        "",
        regex=True
    )
    .str.strip()
)


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


# 연령대
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


# ==========================================
# 7. 행정구역 / 연도 / 성별 선택
# ==========================================
st.subheader("🔎 분석 조건 선택")

col1, col2, col3 = st.columns(3)


with col1:

    districts = (
        df["행정구역명"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_district = st.selectbox(
        "🏙️ 행정구 선택",
        districts
    )


with col2:

    selected_year = st.selectbox(
        "📅 연도 선택",
        available_years,
        index=len(available_years) - 1,
        format_func=lambda x: f"{x}년"
    )


with col3:

    selected_gender = st.selectbox(
        "👥 성별 선택",
        ["전체", "남성", "여성"]
    )


# ==========================================
# 8. 성별별 컬럼 선택
# ==========================================
if selected_gender == "전체":

    prefix = f"{selected_year}년_거주자_"

elif selected_gender == "남성":

    prefix = f"{selected_year}년_남_거주자_"

else:

    prefix = f"{selected_year}년_여_거주자_"


# 선택한 행정구역
selected_data = df[
    df["행정구역명"] == selected_district
]


if selected_data.empty:

    st.error("선택한 행정구역 데이터가 없습니다.")
    st.stop()


selected_row = selected_data.iloc[0]


# ==========================================
# 9. 연령별 인구 데이터 생성
# ==========================================
chart_rows = []
missing_columns = []

for age in age_groups:

    expected_column = normalize_column(
        prefix + age
    )

    actual_column = column_lookup.get(
        expected_column
    )

    if actual_column is None:

        missing_columns.append(prefix + age)
        continue

    value = selected_row[actual_column]

    chart_rows.append({
        "나이": age,
        "인구수": (
            0 if pd.isna(value)
            else int(value)
        )
    })


if not chart_rows:

    st.error(
        "선택한 연도와 성별의 연령별 인구 컬럼을 찾을 수 없습니다."
    )

    st.write("찾으려던 컬럼:")
    st.code(
        "\n".join(prefix + age for age in age_groups)
    )

    st.write("실제 컬럼명:")
    st.code(
        "\n".join(str(col) for col in df.columns[:40])
    )

    st.stop()


if missing_columns:

    st.warning(
        "일부 연령대 데이터가 없어 그래프에서 제외했습니다."
    )


chart_df = pd.DataFrame(chart_rows)


# ==========================================
# 10. 인구 분석 요약
# ==========================================
total_population = int(
    chart_df["인구수"].sum()
)

max_age_row = chart_df.loc[
    chart_df["인구수"].idxmax()
]


st.divider()

st.subheader("📌 인구 분석 요약")

m1, m2, m3 = st.columns(3)


m1.metric(
    "선택한 행정구",
    selected_district
)

m2.metric(
    "연령별 인구 합계",
    f"{total_population:,}명"
)

m3.metric(
    "인구가 가장 많은 연령대",
    max_age_row["나이"],
    f"{int(max_age_row['인구수']):,}명"
)


# ==========================================
# 11. 초록색 꺾은선 그래프
# ==========================================
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


# 선과 점 스타일
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


# 세로축 최대값
max_population = int(
    chart_df["인구수"].max()
)

y_axis_max = max(
    1000,
    ((max_population // 1000) + 1) * 1000
)


# 그래프 레이아웃
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


st.plotly_chart(
    fig,
    use_container_width=True
)


# ==========================================
# 12. 연령별 인구 데이터 표
# ==========================================
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


# ==========================================
# 13. 하단 안내
# ==========================================
st.divider()

st.caption(
    "📊 데이터 출처: population (1).csv"
)

st.caption(
    "한글 인코딩: CP949 / UTF-8 자동 대응"
)
