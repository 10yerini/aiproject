
import csv
from pathlib import Path

import streamlit as st


# ==========================================
# 1. 기본 설정
# ==========================================
st.set_page_config(
    page_title="세계 MBTI 국가별 분석",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 세계 MBTI 국가별 분석")
st.write("국가를 선택하고 MBTI 유형별 비율을 확인해 보세요!")

MBTI_TYPES = [
    "INFJ", "ISFJ", "INTP", "ISFP",
    "ENTP", "INFP", "ENTJ", "ISTP",
    "INTJ", "ESFP", "ESTJ", "ENFP",
    "ESTP", "ISTJ", "ENFJ", "ESFJ"
]

CSV_PATH = Path(__file__).parent / "countriesMBTI_16types.csv"


# ==========================================
# 2. 데이터 불러오기
# ==========================================
@st.cache_data
def load_data():

    data = []

    with open(
        CSV_PATH,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            item = {
                "Country": row["Country"]
            }

            for mbti in MBTI_TYPES:
                item[mbti] = float(row[mbti])

            data.append(item)

    return data


if not CSV_PATH.exists():
    st.error(
        "countriesMBTI_16types.csv 파일을 찾을 수 없습니다."
    )
    st.stop()

try:
    data = load_data()

except Exception as error:
    st.error(f"데이터 오류: {error}")
    st.stop()


# ==========================================
# 3. 국가 선택
# ==========================================
countries = sorted(
    [row["Country"] for row in data]
)

st.sidebar.title("🌎 국가 선택")

selected_country = st.sidebar.selectbox(
    "분석할 국가를 선택하세요.",
    countries,
    index=(
        countries.index("South Korea")
        if "South Korea" in countries
        else 0
    )
)

country_data = next(
    row for row in data
    if row["Country"] == selected_country
)


# ==========================================
# 4. MBTI 비율 순위 계산
# ==========================================
ranking = sorted(
    [
        (mbti, country_data[mbti])
        for mbti in MBTI_TYPES
    ],
    key=lambda x: x[1],
    reverse=True
)

top_mbti = ranking[0][0]
top_value = ranking[0][1]


# ==========================================
# 5. 주요 정보 표시
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
    f"{top_value * 100:.2f}%"
)

st.divider()


# ==========================================
# 6. 그래프 제목
# ==========================================
st.subheader(
    f"📊 {selected_country}의 MBTI 비율"
)

st.caption(
    "1위는 빨간색, 나머지는 파란색 계열의 "
    "그라데이션으로 표시됩니다."
)


# ==========================================
# 7. 그래프 색상 설정
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


# ==========================================
# 8. 막대그래프 만들기
# ==========================================
chart_html = """
<style>

.mbti-chart {
    width: 100%;
    font-family: Arial, sans-serif;
}

.mbti-row {
    display: grid;
    grid-template-columns: 60px 1fr 80px;
    align-items: center;
    gap: 12px;
    margin: 12px 0;
}

.mbti-label {
    font-size: 14px;
    font-weight: bold;
}

.mbti-track {
    width: 100%;
    height: 25px;
    background: rgba(128,128,128,0.12);
    border-radius: 6px;
    overflow: hidden;
}

.mbti-bar {
    height: 25px;
    border-radius: 6px;
}

.mbti-value {
    text-align: right;
    font-size: 13px;
    font-weight: bold;
}

</style>

<div class="mbti-chart">
"""

max_value = max(
    value for mbti, value in ranking
)

for index, (mbti, value) in enumerate(ranking):

    if index == 0:
        bar_color = "#E53935"

    else:
        bar_color = blue_gradient[index - 1]

    width = (
        value / max_value * 100
        if max_value > 0
        else 0
    )

    percentage = value * 100

    chart_html += f"""
    <div class="mbti-row">

        <div class="mbti-label">
            {mbti}
        </div>

        <div class="mbti-track">

            <div
                class="mbti-bar"
                style="
                    width:{width:.2f}%;
                    background:{bar_color};
                "
            >
            </div>

        </div>

        <div class="mbti-value">
            {percentage:.2f}%
        </div>

    </div>
    """

chart_html += "</div>"

st.markdown(
    chart_html,
    unsafe_allow_html=True
)


# ==========================================
# 9. MBTI 순위표
# ==========================================
st.divider()

st.subheader("🏆 MBTI 순위")

for index, (mbti, value) in enumerate(
    ranking,
    start=1
):

    st.write(
        f"**{index}위. {mbti}** "
        f"- {value * 100:.2f}%"
    )


# ==========================================
# 10. 하단 안내
# ==========================================
st.divider()

st.caption(
    "※ 본 자료는 제공된 CSV 데이터의 국가별 MBTI "
    "분포를 시각화한 것입니다."
)

st.caption(
    "※ 개인의 성격을 판단하거나 예측하는 자료는 아닙니다."
)
