
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

# ==========================================
# 1. 페이지 설정
# ==========================================
st.set_page_config(
    page_title="월별 평균기온 예측",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 월별 평균기온 예측 프로그램")
st.write(
    "과거 기온 데이터를 분석하고 "
    "선형회귀 알고리즘으로 미래의 기온을 예측해 봐요!"
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
        df = pd.read_csv(file_path, encoding="cp949")
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding="utf-8-sig")

except Exception as e:
    st.error(f"CSV 파일을 읽을 수 없습니다: {e}")
    st.stop()

df.columns = df.columns.str.strip()

# ==========================================
# 3. 데이터 전처리
# ==========================================
required_columns = ["일시", "평균기온(℃)"]

for col in required_columns:
    if col not in df.columns:
        st.error(f"'{col}' 컬럼이 없습니다.")
        st.stop()

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

# 날짜에서 두 자리 연도와 월 추출
short_years = []
months = []

for value in df["일시"]:
    try:
        month_name, year_text = str(value).strip().split("-")

        short_years.append(int(year_text))
        months.append(month_dict[month_name])

    except (ValueError, KeyError, IndexError):
        short_years.append(None)
        months.append(None)

df["연도_두자리"] = short_years
df["월"] = months

# ==========================================
# 4. 실제 연도 계산
# 1907년부터 2026년까지 자동 변환
# ==========================================
century = 1900
previous_year = None
full_years = []

for year in df["연도_두자리"]:

    if year is None:
        full_years.append(None)
        continue

    # 99년에서 00년으로 넘어가는 경우
    if (
        previous_year is not None
        and previous_year - year > 50
    ):
        century += 100

    full_years.append(century + year)
    previous_year = year

df["연도"] = full_years

# 기온 숫자로 변환
df["평균기온(℃)"] = pd.to_numeric(
    df["평균기온(℃)"],
    errors="coerce"
)

# 결측값 제거
df = df.dropna(
    subset=["연도", "월", "평균기온(℃)"]
)

df["연도"] = df["연도"].astype(int)
df["월"] = df["월"].astype(int)

# 같은 연도와 월에 데이터가 여러 개라면 평균 계산
df = (
    df.groupby(["연도", "월"], as_index=False)
    ["평균기온(℃)"].mean()
)

df = df.sort_values(["연도", "월"])

# ==========================================
# 5. 데이터 범위 표시
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
        "분석 데이터 개수",
        f"{len(df):,}개"
    )

st.divider()

# ==========================================
# 6. 사용자 입력
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
# 7. 선택한 월의 데이터 추출
# ==========================================
monthly_df = df[
    df["월"] == selected_month
].sort_values("연도")

if len(monthly_df) < 2:
    st.error(
        "선택한 월의 데이터가 2개 미만입니다."
    )
    st.stop()

years = monthly_df["연도"].to_numpy(dtype=float)

temperatures = monthly_df[
    "평균기온(℃)"
].to_numpy(dtype=float)

# ==========================================
# 8. 선형회귀 알고리즘
# ==========================================
# y = ax + b
#
# x : 연도
# y : 평균기온
# a : 기울기
# b : 절편

slope, intercept = np.polyfit(
    years,
    temperatures,
    1
)

# 입력한 연도의 기온 예측
predicted_temp = (
    slope * selected_year + intercept
)

# 과거 데이터에 대한 회귀선 예측값
fitted_values = (
    slope * years + intercept
)

# ==========================================
# 9. 결정계수 R² 계산
# ==========================================
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

# ==========================================
# 10. 예측 결과 표시
# ==========================================
st.divider()

st.subheader("🔮 예측 결과")

result1, result2, result3 = st.columns(3)

with result1:
    st.metric(
        "예측 연도",
        f"{selected_year}년 {selected_month}월"
    )

with result2:
    st.metric(
        "예측 평균기온",
        f"{predicted_temp:.1f} ℃"
    )

with result3:
    st.metric(
        "결정계수 R²",
        f"{r_squared:.3f}"
    )

if selected_year > int(years.max()):

    st.info(
        "🔮 과거 데이터의 추세를 바탕으로 "
        "미래의 평균기온을 예측했습니다."
    )

elif selected_year < int(years.min()):

    st.info(
        "입력한 연도는 과거 데이터보다 이전이므로 "
        "회귀선을 이용한 과거 방향의 추정값입니다."
    )

else:

    st.info(
        "입력한 연도는 실제 관측 데이터 범위에 있습니다. "
        "회귀선을 이용한 추정값입니다."
    )

# ==========================================
# 11. 그래프 시각화
# ==========================================
st.divider()

st.subheader(
    f"📈 {selected_month}월 연도별 평균기온 분석"
)

fig = go.Figure()

# ------------------------------------------
# 실제 평균기온
# ------------------------------------------
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
        "실제 평균기온: %{y:.1f}℃"
        "<extra></extra>"
    )
)

# ------------------------------------------
# 선형회귀 추세선
# ------------------------------------------
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

line_temperatures = (
    slope * line_years + intercept
)

fig.add_trace(
    go.Scatter(
        x=line_years,
        y=line_temperatures,
        mode="lines",
        name="선형회귀 추세선",
        line=dict(
            width=3,
            dash="dash"
        ),
        hovertemplate=
        "연도: %{x:.0f}년<br>"
        "회귀선: %{y:.1f}℃"
        "<extra></extra>"
    )
)

# ------------------------------------------
# 실제 기온과 회귀선 사이의 잔차
# ------------------------------------------
residual_x = []
residual_y = []

for x, actual, fitted in zip(
    years,
    temperatures,
    fitted_values
):

    residual_x.extend([x, x, None])

    residual_y.extend([
        actual,
        fitted,
        None
    ])

fig.add_trace(
    go.Scatter(
        x=residual_x,
        y=residual_y,
        mode="lines",
        name="잔차(오차)",
        line=dict(
            width=1,
            dash="dot"
        ),
        hoverinfo="skip"
    )
)

# ------------------------------------------
# 미래 예측 기온
# ------------------------------------------
fig.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[predicted_temp],
        mode="markers+text",
        name="예측 평균기온",
        marker=dict(
            size=16,
            symbol="star"
        ),
        text=[f"{predicted_temp:.1f}℃"],
        textposition="top center",
        hovertemplate=
        "예측 연도: %{x:.0f}년<br>"
        "예측 기온: %{y:.1f}℃"
        "<extra></extra>"
    )
)

# 그래프 디자인
fig.update_layout(
    title=f"{selected_month}월 평균기온 선형회귀 분석",
    xaxis_title="연도",
    yaxis_title="평균기온 (℃)",
    height=600,
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0
    )
)

fig.update_xaxes(
    dtick=10
)

fig.update_yaxes(
    zeroline=True
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================
# 12. 그래프 읽는 방법
# ==========================================
st.subheader("🎨 그래프 읽는 방법")

st.markdown("""
- **실제 평균기온**: 과거에 실제로 측정한 기온입니다.
- **선형회귀 추세선**: 컴퓨터가 과거 데이터를 학습해서 찾은 직선입니다.
- **잔차(오차)**: 실제 기온과 회귀선이 예측한 기온 사이의 차이입니다.
- **예측 평균기온**: 사용자가 입력한 연도의 예상 기온입니다.
""")

st.divider()

# ==========================================
# 13. 선형회귀 분석 결과
# ==========================================
st.subheader("🧮 선형회귀 분석 결과")

st.write("컴퓨터가 계산한 선형회귀 방정식입니다.")

st.latex(
    rf"\hat{{y}} = {slope:.4f}x + ({intercept:.4f})"
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "기울기",
        f"{slope:.4f}"
    )

with col2:
    st.metric(
        "절편",
        f"{intercept:.4f}"
    )

with col3:
    st.metric(
        "결정계수 R²",
        f"{r_squared:.4f}"
    )

if slope > 0:

    st.success(
        f"📈 {selected_month}월 평균기온은 "
        f"분석 기간 동안 연평균 약 "
        f"{slope:.3f}℃ 상승하는 추세입니다."
    )

elif slope < 0:

    st.info(
        f"📉 {selected_month}월 평균기온은 "
        f"분석 기간 동안 연평균 약 "
        f"{abs(slope):.3f}℃ 하락하는 추세입니다."
    )

else:

    st.info(
        "평균기온의 장기적인 변화 추세가 거의 없습니다."
    )

# ==========================================
# 14. 선형회귀 알고리즘 설명
# ==========================================
st.divider()

st.subheader("📚 선형회귀 알고리즘 쉽게 이해하기")

st.markdown("""
### ① 선형회귀란?

선형회귀는 여러 데이터의 관계를 가장 잘 나타내는
직선 하나를 찾는 머신러닝 알고리즘이에요.

이번 프로그램에서는 과거의 연도와 평균기온 사이의
관계를 학습합니다.

### ② 직선은 어떻게 만들어질까요?

선형회귀는 다음과 같은 직선의 방정식을 사용합니다.
""")

st.latex(
    r"\hat{y} = ax + b"
)

st.markdown("""
- x : 연도
- ŷ : 예측한 평균기온
- a : 기울기
- b : 절편

컴퓨터는 실제 기온과 직선이 예측한 기온의 차이를 계산합니다.

### ③ 가장 잘 맞는 직선은 어떻게 찾을까요?

실제 데이터 점과 직선 사이의 차이를 잔차라고 합니다.

선형회귀는 잔차를 제곱한 뒤 모두 더한 값이
가장 작아지도록 직선을 결정합니다.

이를 최소제곱법이라고 해요.

그래프에서 실제 기온 점과 추세선 사이에 표시된
세로 점선이 잔차를 나타냅니다.

### ④ 미래 기온은 어떻게 예측할까요?

컴퓨터가 과거 데이터로 직선을 완성하면
새로운 연도를 직선에 대입합니다.

예를 들어 2030년을 입력하면
2030이라는 숫자를 직선의 x에 넣어
예상 평균기온을 계산하는 방식입니다.

### ⑤ 결정계수 R²란?

R²는 만들어진 회귀선이 과거 데이터의 변화를
얼마나 잘 설명하는지 나타내는 값입니다.

- 1에 가까우면 과거 데이터에 잘 맞습니다.
- 0에 가까우면 과거 데이터의 변화를 잘 설명하지 못합니다.
- 음수가 나올 수도 있습니다.

R²가 높다고 미래 기온까지 정확하게 예측한다는 뜻은 아닙니다.

### ⑥ 이 프로그램의 한계

기온은 여러 가지 자연환경의 영향을 받습니다.

- 온실가스
- 해류
- 기압
- 강수량
- 계절과 기후 변화

하지만 이 프로그램은 연도와 평균기온 두 가지 관계만
이용하는 단순 선형회귀 모델입니다.

따라서 실제 일기예보가 아니라
과거의 장기적인 기온 변화 추세를 이해하기 위한
학습용 프로그램입니다.
""")

# ==========================================
# 15. 마무리
# ==========================================
st.divider()

st.success(
    "🌍 과거 기온 데이터를 이용한 "
    "선형회귀 분석과 미래 기온 예측이 완료되었습니다!"
)

st.caption(
    "기온 데이터 분석 학습 프로그램 | "
    "Linear Regression Temperature Prediction"
)
