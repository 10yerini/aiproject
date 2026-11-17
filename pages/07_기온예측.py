
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

# =====================================
# 1. 기본 페이지 설정
# =====================================
st.set_page_config(
    page_title="월별 평균기온 예측",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 월별 평균기온 예측 프로그램")
st.write("과거 기온 데이터를 이용해 미래의 평균기온을 예측해 봐요!")
st.divider()

# =====================================
# 2. CSV 파일 불러오기
# =====================================
file_paths = [
    Path(__file__).resolve().parent.parent / "temperature.csv",
    Path(__file__).resolve().parent / "temperature.csv"
]

file_path = next(
    (p for p in file_paths if p.exists()),
    None
)

if file_path is None:
    st.error(
        "temperature.csv 파일을 찾을 수 없습니다. "
        "pages 폴더의 상위 폴더에 CSV 파일을 넣어 주세요."
    )
    st.stop()

try:
    df = pd.read_csv(file_path, encoding="cp949")
except Exception as e:
    st.error(f"CSV 파일을 읽는 중 오류가 발생했습니다: {e}")
    st.stop()

# =====================================
# 3. 데이터 전처리
# =====================================
df.columns = df.columns.str.strip()

required_columns = ["일시", "평균기온(℃)"]

for col in required_columns:
    if col not in df.columns:
        st.error(f"CSV 파일에 '{col}' 컬럼이 없습니다.")
        st.stop()

# 영문 월 이름을 숫자로 변환
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

def convert_date(value):
    try:
        month_name, year_text = str(value).strip().split("-")
        month = month_dict[month_name]
        year = 2000 + int(year_text)
        return pd.Series([year, month])
    except (ValueError, KeyError):
        return pd.Series([np.nan, np.nan])

df[["연도", "월"]] = df["일시"].apply(convert_date)

df["평균기온(℃)"] = pd.to_numeric(
    df["평균기온(℃)"],
    errors="coerce"
)

df = df.dropna(
    subset=["연도", "월", "평균기온(℃)"]
)

df["연도"] = df["연도"].astype(int)
df["월"] = df["월"].astype(int)

# =====================================
# 4. 사용자 입력
# =====================================
st.subheader("📌 예측 조건 선택")

col1, col2 = st.columns(2)

with col1:
    selected_month = st.selectbox(
        "예측할 월을 선택하세요",
        list(range(1, 13)),
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

# =====================================
# 5. 선택한 월의 데이터 추출
# =====================================
monthly_df = df[
    df["월"] == selected_month
].sort_values("연도")

if len(monthly_df) < 2:
    st.error("선택한 월의 데이터가 2개 미만이어서 예측할 수 없습니다.")
    st.stop()

years = monthly_df["연도"].to_numpy(dtype=float)
temperatures = monthly_df["평균기온(℃)"].to_numpy(dtype=float)

# =====================================
# 6. 선형회귀 알고리즘
# =====================================
# y = ax + b
# a : 기울기
# b : y절편

slope, intercept = np.polyfit(
    years,
    temperatures,
    1
)

# 예측 연도의 기온 계산
predicted_temp = (
    slope * selected_year + intercept
)

# 회귀선의 예측값
fitted_values = (
    slope * years + intercept
)

# 결정계수 R² 계산
ss_res = np.sum(
    (temperatures - fitted_values) ** 2
)

ss_tot = np.sum(
    (temperatures - np.mean(temperatures)) ** 2
)

if ss_tot == 0:
    r_squared = 0
else:
    r_squared = 1 - (ss_res / ss_tot)

# =====================================
# 7. 예측 결과 표시
# =====================================
st.divider()
st.subheader("🔮 기온 예측 결과")

result1, result2, result3 = st.columns(3)

with result1:
    st.metric(
        label="예측 연도",
        value=f"{selected_year}년 {selected_month}월"
    )

with result2:
    st.metric(
        label="예측 평균기온",
        value=f"{predicted_temp:.1f} ℃"
    )

with result3:
    st.metric(
        label="결정계수 (R²)",
        value=f"{r_squared:.3f}"
    )

if selected_year > int(years.max()):
    st.info(
        "💡 과거 데이터의 추세를 바탕으로 미래를 예측한 값입니다. "
        "실제 기온과 다를 수 있습니다."
    )
elif selected_year < int(years.min()):
    st.info(
        "💡 입력한 연도가 관측 데이터보다 이전이므로 "
        "과거 방향으로 외삽한 예측값입니다."
    )
else:
    st.info(
        "💡 입력한 연도는 관측 데이터 범위 안에 있습니다. "
        "회귀선을 이용한 추정값입니다."
    )

# =====================================
# 8. 그래프 만들기
# =====================================
st.divider()
st.subheader(
    f"📈 {selected_month}월 연도별 평균기온 및 선형회귀 예측"
)

fig = go.Figure()

# 실제 관측 기온
fig.add_trace(
    go.Scatter(
        x=years,
        y=temperatures,
        mode="lines+markers",
        name="실제 평균기온",
        line=dict(width=2),
        marker=dict(size=7),
        hovertemplate=
        "연도: %{x:.0f}년<br>" +
        "실제 기온: %{y:.1f}℃<extra></extra>"
    )
)

# 회귀선은 관측 기간과 예측 연도를 모두 포함
line_start = min(float(years.min()), float(selected_year))
line_end = max(float(years.max()), float(selected_year))

line_years = np.linspace(
    line_start,
    line_end,
    200
)

line_temps = slope * line_years + intercept

fig.add_trace(
    go.Scatter(
        x=line_years,
        y=line_temps,
        mode="lines",
        name="선형회귀 추세선",
        line=dict(width=3, dash="dash"),
        hovertemplate=
        "연도: %{x:.0f}년<br>" +
        "회귀선 기온: %{y:.1f}℃<extra></extra>"
    )
)

# 잔차(실제 기온과 회귀선의 차이)를 세로선으로 표시
residual_x = []
residual_y = []

for x, actual, fitted in zip(
    years, temperatures, fitted_values
):
    residual_x.extend([x, x, None])
    residual_y.extend([actual, fitted, None])

fig.add_trace(
    go.Scatter(
        x=residual_x,
        y=residual_y,
        mode="lines",
        name="잔차(오차)",
        line=dict(width=1, dash="dot"),
        hoverinfo="skip"
    )
)

# 예측 기온 표시
fig.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[predicted_temp],
        mode="markers+text",
        name="예측 기온",
        marker=dict(size=15, symbol="star"),
        text=[f"{predicted_temp:.1f}℃"],
        textposition="top center",
        hovertemplate=
        "예측 연도: %{x}년<br>" +
        "예측 기온: %{y:.1f}℃<extra></extra>"
    )
)

fig.update_layout(
    title=f"{selected_month}월 평균기온 선형회귀 분석",
    xaxis_title="연도",
    yaxis_title="평균기온 (℃)",
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0
    ),
    height=600
)

fig.update_xaxes(dtick=1)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================
# 9. 선형회귀 분석 결과 설명
# =====================================
st.divider()
st.subheader("🧮 선형회귀 분석 결과")

st.write("**선형회귀 계산식**")

st.latex(
    rf"\hat{{y}} = {slope:.4f}x + ({intercept:.4f})"
)

st.write(f"- 기울기: {slope:.4f}")
st.write(f"- 절편: {intercept:.4f}")
st.write(f"- 결정계수 R²: {r_squared:.4f}")

if slope > 0:
    st.success(
        f"📈 분석 결과, {selected_month}월의 평균기온은 "
        f"연평균 약 {slope:.3f}℃씩 상승하는 추세입니다."
    )
elif slope < 0:
    st.info(
        f"📉 분석 결과, {selected_month}월의 평균기온은 "
        f"연평균 약 {abs(slope):.3f}℃씩 하락하는 추세입니다."
    )
else:
    st.info("분석 결과, 평균기온의 장기적인 변화 추세가 거의 없습니다.")

# =====================================
# 10. 선형회귀 알고리즘 쉽게 이해하기
# =====================================
st.divider()
st.subheader("📚 선형회귀 알고리즘이란?")

st.markdown("""
### ① 선형회귀란 무엇일까요?

선형회귀는 여러 개의 데이터 점을 가장 잘 설명하는
**하나의 직선**을 찾는 인공지능 학습 방법이에요.

이번 프로그램에서는 다음과 같이 생각하면 됩니다.

- 가로축(X): 연도
- 세로축(Y): 해당 월의 평균기온
- 점: 실제로 관측된 평균기온
- 점선: 선형회귀가 찾아낸 추세선
- 별 모양: 미래의 예측 기온

### ② 컴퓨터는 어떻게 직선을 찾을까요?

컴퓨터는 실제 기온과 직선이 예측한 기온의 차이를 계산해요.

이 차이를 잔차(오차)라고 합니다.

그래프에서 세로로 표시된 점선이 바로 이 차이예요.

선형회귀는 모든 잔차의 제곱을 더한 값이
가장 작아지도록 직선의 기울기와 절편을 결정합니다.

### ③ 미래의 기온은 어떻게 예측할까요?

과거의 연도와 평균기온 사이에서 발견한 관계를 이용해
입력한 미래 연도를 직선에 대입합니다.

그러면 해당 연도의 예상 평균기온이 계산됩니다.

### ④ 결정계수 R²는 무엇일까요?

R²는 회귀선이 과거 기온 데이터의 변화를
얼마나 잘 설명하는지를 나타내는 지표예요.

- 1에 가까울수록 데이터에 잘 맞습니다.
- 0에 가까울수록 데이터의 변화를 잘 설명하지 못합니다.
- 1보다 작을 수도 있고, 음수가 나올 수도 있습니다.

단, R²가 높더라도 미래의 기온을 정확하게 맞힌다는 보장은 없어요.

### ⑤ 꼭 기억하세요!

기온은 온실가스, 해류, 기압, 강수량 등
다양한 요인의 영향을 받아요.

이 프로그램은 과거의 연도별 평균기온만 이용하는
단순 선형회귀 모델이므로 실제 날씨 예보가 아닌
장기적인 기온 추세를 살펴보는 학습용 프로그램입니다.
""")

st.divider()

st.caption(
    "🌍 기상 데이터 분석 학습 프로그램 | "
    "Linear Regression Temperature Prediction"
)
