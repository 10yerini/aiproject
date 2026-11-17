
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


# ==========================================
# 1. 페이지 설정
# ==========================================

st.set_page_config(
    page_title="AI 기온 예측 및 알고리즘 비교",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ AI 월별 평균기온 예측 및 비교")
st.write(
    "3가지 머신러닝 알고리즘으로 미래 기온을 예측하고 "
    "결과를 비교해 보세요!"
)

st.divider()


# ==========================================
# 2. CSV 파일 불러오기
# ==========================================

file_path = (
    Path(__file__).resolve().parent.parent
    / "temperature.csv"
)

if not file_path.exists():
    st.error(
        "temperature.csv 파일을 찾을 수 없습니다.\n\n"
        "pages 폴더의 상위 폴더에 "
        "temperature.csv 파일을 넣어 주세요."
    )
    st.stop()

try:
    try:
        df = pd.read_csv(
            file_path,
            encoding="cp949"
        )
    except UnicodeDecodeError:
        df = pd.read_csv(
            file_path,
            encoding="utf-8-sig"
        )

except Exception as e:
    st.error(f"CSV 파일을 읽을 수 없습니다: {e}")
    st.stop()

df.columns = df.columns.str.strip()


# ==========================================
# 3. 데이터 확인
# ==========================================

required_columns = [
    "일시",
    "평균기온(℃)"
]

for col in required_columns:
    if col not in df.columns:
        st.error(f"'{col}' 컬럼이 없습니다.")
        st.stop()


# ==========================================
# 4. 날짜 및 연도 전처리
# ==========================================

month_dict = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12
}

short_years = []
months = []

for value in df["일시"]:

    try:
        month_name, year_text = (
            str(value).strip().split("-")
        )

        short_years.append(int(year_text))
        months.append(month_dict[month_name])

    except (ValueError, KeyError, IndexError):
        short_years.append(None)
        months.append(None)


df["연도_두자리"] = short_years
df["월"] = months


# ==========================================
# 5. 1907년부터 2026년까지 실제 연도 계산
# ==========================================

century = 1900
previous_year = None
full_years = []

for year in df["연도_두자리"]:

    if year is None:
        full_years.append(None)
        continue

    # 99년에서 00년으로 넘어갈 때 세기 변경
    if (
        previous_year is not None
        and previous_year - year > 50
    ):
        century += 100

    full_years.append(century + year)

    previous_year = year


df["연도"] = full_years


# ==========================================
# 6. 기온 데이터 정리
# ==========================================

df["평균기온(℃)"] = pd.to_numeric(
    df["평균기온(℃)"],
    errors="coerce"
)

df = df.dropna(
    subset=[
        "연도",
        "월",
        "평균기온(℃)"
    ]
)

df["연도"] = df["연도"].astype(int)
df["월"] = df["월"].astype(int)


# 같은 연도와 월의 데이터는 평균으로 통합
df = (
    df.groupby(
        ["연도", "월"],
        as_index=False
    )["평균기온(℃)"]
    .mean()
)

df = df.sort_values(
    ["연도", "월"]
)


# ==========================================
# 7. 전체 데이터 정보
# ==========================================

st.subheader("📊 데이터 정보")

info1, info2, info3 = st.columns(3)

with info1:
    st.metric(
        "가장 오래된 연도",
        f"{df['연도'].min()}년"
    )

with info2:
    st.metric(
        "가장 최근 연도",
        f"{df['연도'].max()}년"
    )

with info3:
    st.metric(
        "전체 데이터",
        f"{len(df):,}개"
    )

st.divider()


# ==========================================
# 8. 사용자 입력
# ==========================================

st.subheader("🔍 예측 조건 선택")

col1, col2 = st.columns(2)

with col1:

    selected_month = st.selectbox(
        "예측할 월을 선택하세요",
        range(1, 13),
        format_func=lambda x: f"{x}월"
    )

with col2:

    selected_year = st.number_input(
        "예측할 연도를 입력하세요",
        min_value=1900,
        max_value=2100,
        value=2027,
        step=1
    )

selected_year = int(selected_year)


# ==========================================
# 9. 알고리즘 선택
# ==========================================

st.subheader("🤖 사용할 알고리즘 선택")

algorithm_dict = {
    "선형회귀": "Linear Regression",
    "다항회귀": "Polynomial Regression",
    "랜덤 포레스트": "Random Forest Regression"
}

selected_algorithm = st.selectbox(
    "분석에 사용할 알고리즘을 선택하세요",
    list(algorithm_dict.keys())
)

st.info(
    f"현재 선택한 알고리즘: "
    f"{selected_algorithm} "
    f"({algorithm_dict[selected_algorithm]})"
)


# ==========================================
# 10. 선택한 월의 데이터 추출
# ==========================================

monthly_df = df[
    df["월"] == selected_month
].sort_values("연도")

if len(monthly_df) < 4:
    st.error(
        "선택한 월의 데이터가 부족합니다. "
        "최소 4개 이상의 연도 데이터가 필요합니다."
    )
    st.stop()

years = monthly_df[
    "연도"
].to_numpy(dtype=float)

temperatures = monthly_df[
    "평균기온(℃)"
].to_numpy(dtype=float)

# 연도를 0부터 시작하는 값으로 변환
# 큰 연도 숫자로 인한 계산 문제를 줄임
base_year = years.min()

X = (years - base_year).reshape(-1, 1)

y = temperatures


# ==========================================
# 11. 3가지 머신러닝 모델 만들기
# ==========================================

models = {

    # 첫 번째 알고리즘
    "선형회귀": LinearRegression(),

    # 두 번째 알고리즘
    # 2차 다항식으로 곡선 학습
    "다항회귀": make_pipeline(
        PolynomialFeatures(
            degree=2,
            include_bias=False
        ),
        LinearRegression()
    ),

    # 세 번째 알고리즘
    "랜덤 포레스트": RandomForestRegressor(
        n_estimators=200,
        max_depth=4,
        min_samples_leaf=2,
        random_state=42
    )

}


# ==========================================
# 12. 3가지 알고리즘 학습
# ==========================================

predictions = {}
model_scores = {}

for name, model in models.items():

    model.fit(X, y)

    # 입력한 예측 연도
    future_X = np.array([
        [selected_year - base_year]
    ])

    # 미래 기온 예측
    prediction = model.predict(future_X)[0]

    predictions[name] = prediction

    # 학습 데이터 설명력
    model_scores[name] = model.score(X, y)


# ==========================================
# 13. 선택한 알고리즘 결과
# ==========================================

selected_model = models[selected_algorithm]

selected_prediction = predictions[
    selected_algorithm
]

selected_score = model_scores[
    selected_algorithm
]


# ==========================================
# 14. 세 알고리즘 예측값 한눈에 비교
# ==========================================

st.divider()

st.subheader("🔮 세 가지 알고리즘 예측 결과 비교")

st.write(
    f"**예측 대상: {selected_year}년 {selected_month}월**"
)

result1, result2, result3 = st.columns(3)

with result1:

    st.metric(
        label="① 선형회귀",
        value=f"{predictions['선형회귀']:.2f} ℃"
    )

with result2:

    st.metric(
        label="② 다항회귀",
        value=f"{predictions['다항회귀']:.2f} ℃"
    )

with result3:

    st.metric(
        label="③ 랜덤 포레스트",
        value=f"{predictions['랜덤 포레스트']:.2f} ℃"
    )


# ==========================================
# 15. 예측값 비교 표
# ==========================================

comparison_df = pd.DataFrame({
    "알고리즘": [
        "선형회귀",
        "다항회귀",
        "랜덤 포레스트"
    ],

    "예측 평균기온(℃)": [
        predictions["선형회귀"],
        predictions["다항회귀"],
        predictions["랜덤 포레스트"]
    ],

    "학습 데이터 R²": [
        model_scores["선형회귀"],
        model_scores["다항회귀"],
        model_scores["랜덤 포레스트"]
    ]
})

st.dataframe(
    comparison_df.style.format({
        "예측 평균기온(℃)": "{:.2f}",
        "학습 데이터 R²": "{:.3f}"
    }),
    use_container_width=True,
    hide_index=True
)


# ==========================================
# 16. 예측값 막대그래프
# ==========================================

st.subheader("📊 알고리즘별 예측 기온 비교")

bar_fig = go.Figure()

bar_fig.add_trace(
    go.Bar(
        x=comparison_df["알고리즘"],
        y=comparison_df["예측 평균기온(℃)"],
        text=[
            f"{v:.2f}℃"
            for v in comparison_df["예측 평균기온(℃)"]
        ],
        textposition="outside",
        name="예측 평균기온"
    )
)

bar_fig.update_layout(
    title="세 가지 알고리즘의 미래 기온 예측",
    xaxis_title="알고리즘",
    yaxis_title="예측 평균기온 (℃)",
    height=450,
    showlegend=False
)

st.plotly_chart(
    bar_fig,
    use_container_width=True
)


# ==========================================
# 17. 선택한 알고리즘의 추세선
# ==========================================

st.divider()

st.subheader(
    f"📈 {selected_algorithm} 분석 그래프"
)

# 관측 기간과 예측 연도를 모두 포함
line_start = min(
    float(years.min()),
    float(selected_year)
)

line_end = max(
    float(years.max()),
    float(selected_year)
)

line_years = np.linspace(
    line_start,
    line_end,
    300
)

line_X = (
    line_years - base_year
).reshape(-1, 1)

line_temperatures = selected_model.predict(
    line_X
)

# 실제 관측 기온
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=years,
        y=temperatures,
        mode="lines+markers",
        name="실제 평균기온",
        line=dict(width=2),
        marker=dict(size=6),
        hovertemplate=
        "연도: %{x:.0f}년<br>"
        "실제 기온: %{y:.1f}℃"
        "<extra></extra>"
    )
)

# 선택한 알고리즘의 추세선
line_shape = (
    "linear"
    if selected_algorithm in ["선형회귀", "다항회귀"]
    else "hv"
)

fig.add_trace(
    go.Scatter(
        x=line_years,
        y=line_temperatures,
        mode="lines",
        name=f"{selected_algorithm} 추세선",
        line=dict(
            width=3,
            dash="dash",
            shape=line_shape
        ),
        hovertemplate=
        "연도: %{x:.0f}년<br>"
        "예측 기온: %{y:.1f}℃"
        "<extra></extra>"
    )
)

# 선택한 예측 기온
fig.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[selected_prediction],
        mode="markers+text",
        name="선택 알고리즘 예측값",
        marker=dict(
            size=17,
            symbol="star"
        ),
        text=[
            f"{selected_prediction:.2f}℃"
        ],
        textposition="top center"
    )
)

fig.update_layout(
    title=(
        f"{selected_month}월 "
        f"{selected_algorithm} 분석"
    ),
    xaxis_title="연도",
    yaxis_title="평균기온 (℃)",
    height=600,
    hovermode="x unified",
    legend=dict(
        orientation="h",
        y=1.02,
        x=0
    )
)

fig.update_xaxes(dtick=10)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ==========================================
# 18. 선택한 알고리즘의 상세 결과
# ==========================================

st.subheader("🧮 선택한 알고리즘의 분석 결과")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "선택한 알고리즘",
        selected_algorithm
    )

with col2:

    st.metric(
        "예측 평균기온",
        f"{selected_prediction:.2f} ℃"
    )

with col3:

    st.metric(
        "학습 데이터 R²",
        f"{selected_score:.3f}"
    )

st.caption(
    "R²는 학습에 사용한 과거 데이터에 대한 설명력입니다. "
    "미래 예측 정확도를 직접 나타내는 값은 아닙니다."
)


# ==========================================
# 19. 알고리즘별 작동 원리 설명
# ==========================================

st.divider()

st.subheader("📚 인공지능 알고리즘 쉽게 이해하기")

with st.expander("① 선형회귀 알고리즘", expanded=True):

    st.markdown("""
    **어떻게 작동할까요?**

    과거의 연도와 평균기온을 이용해
    가장 잘 맞는 직선 하나를 찾습니다.

    직선의 기울기를 이용해 미래의 기온을 예측합니다.

    **특징**
    - 전체적인 상승과 하락 추세를 파악합니다.
    - 계산 방법이 간단합니다.
    - 일정한 직선 추세를 가정합니다.
    """)

    st.latex(
        r"\hat{y} = ax+b"
    )

with st.expander("② 다항회귀 알고리즘"):

    st.markdown("""
    **어떻게 작동할까요?**

    선형회귀는 직선을 사용하지만
    다항회귀는 곡선을 이용합니다.

    이번 프로그램은 2차 다항식을 사용합니다.

    따라서 기온이 일정하게 상승하거나 하락하지 않고
    휘어지는 변화도 학습할 수 있습니다.

    **특징**
    - 곡선 형태의 변화를 학습합니다.
    - 직선으로 표현하기 어려운 패턴을 찾을 수 있습니다.
    - 데이터에 지나치게 맞춰질 수 있습니다.
    """)

    st.latex(
        r"\hat{y}=ax^2+bx+c"
    )

with st.expander("③ 랜덤 포레스트 알고리즘"):

    st.markdown("""
    **어떻게 작동할까요?**

    랜덤 포레스트는 여러 개의 결정나무를 만듭니다.

    각각의 나무가 과거 기온 데이터를 학습하고
    각자 예측한 결과를 계산합니다.

    마지막에는 여러 나무의 예측값을 평균 내어
    최종 예측 기온을 결정합니다.

    **특징**
    - 여러 결정나무의 결과를 결합합니다.
    - 복잡한 데이터 패턴을 학습할 수 있습니다.
    - 학습 데이터 범위를 벗어난 미래 예측에는 한계가 있습니다.
    """)

    st.latex(
        r"\hat{y}=\frac{1}{N}\sum_{i=1}^{N}T_i(x)"
    )


# ==========================================
# 20. 세 알고리즘 비교 설명
# ==========================================

st.divider()

st.subheader("🔍 세 가지 알고리즘 비교하기")

explain_df = pd.DataFrame({

    "알고리즘": [
        "선형회귀",
        "다항회귀",
        "랜덤 포레스트"
    ],

    "학습 방식": [
        "직선",
        "곡선",
        "여러 결정나무"
    ],

    "장점": [
        "전체적인 추세 파악",
        "곡선 형태의 변화 학습",
        "복잡한 패턴 학습"
    ],

    "주의할 점": [
        "직선 추세만 표현",
        "과적합 가능성",
        "미래 외삽에 한계"
    ]
})

st.dataframe(
    explain_df,
    use_container_width=True,
    hide_index=True
)


# ==========================================
# 21. 최종 안내
# ==========================================

st.divider()

st.success(
    "🎉 3가지 알고리즘의 기온 예측과 비교가 완료되었습니다!"
)

st.markdown("""
### 💡 생각해 볼 문제

1. 세 가지 알고리즘 중 예측 기온이 가장 높은 것은 무엇일까요?
2. 세 가지 알고리즘의 예측값은 왜 서로 다를까요?
3. 과거 데이터에 가장 잘 맞는 알고리즘이 미래도 가장 잘 예측할까요?
4. 어떤 알고리즘이 기온의 장기적인 변화를 설명하기에 적합할까요?

직접 월과 예측 연도를 바꾸면서 결과를 비교해 보세요!
""")

st.caption(
    "🌍 AI 기온 예측 및 알고리즘 비교 학습 프로그램"
)
