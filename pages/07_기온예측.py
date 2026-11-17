
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# ---------------------------------
# 1. 페이지 설정
# ---------------------------------
st.set_page_config(
    page_title="월별 평균기온 예측",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 월별 평균기온 분석 및 예측")
st.write(
    "과거 기온 데이터를 분석하고 "
    "선형회귀 알고리즘으로 미래의 평균기온을 예측해 보세요!"
)

st.divider()

# ---------------------------------
# 2. CSV 파일 불러오기
# ---------------------------------
file_name = "tmeperature.csv"

current_dir = Path(__file__).resolve().parent

possible_paths = [
    current_dir / file_name,
    current_dir.parent / file_name,
    Path.cwd() / file_name
]

file_path = next(
    (p for p in possible_paths if p.exists()),
    None
)

if file_path is None:
    st.error(
        f"{file_name} 파일을 찾을 수 없습니다. "
        "app.py가 있는 폴더 또는 상위 폴더에 CSV 파일을 넣어 주세요."
    )
    st.stop()

df = None

for encoding in ["cp949", "utf-8-sig", "euc-kr"]:
    try:
        df = pd.read_csv(file_path, encoding=encoding)
        break
    except UnicodeDecodeError:
        continue

if df is None:
    st.error("CSV 파일의 인코딩을 읽을 수 없습니다.")
    st.stop()

# ---------------------------------
# 3. 데이터 전처리
# ---------------------------------
required_columns = ["일시", "평균기온(℃)"]

if not all(col in df.columns for col in required_columns):
    st.error(
        "CSV 파일에 '일시' 또는 '평균기온(℃)' 컬럼이 없습니다."
    )
    st.write("현재 CSV 컬럼:", df.columns.tolist())
    st.stop()

df = df[required_columns].copy()

# 날짜 변환
df["날짜"] = pd.to_datetime(
    df["일시"].astype(str).str.strip(),
    format="%b-%y",
    errors="coerce"
)

df["평균기온(℃)"] = pd.to_numeric(
    df["평균기온(℃)"],
    errors="coerce"
)

df = df.dropna(
    subset=["날짜", "평균기온(℃)"]
).copy()

df["연도"] = df["날짜"].dt.year
df["월"] = df["날짜"].dt.month

if df.empty:
    st.error("분석할 수 있는 기온 데이터가 없습니다.")
    st.stop()

# 동일한 연월 데이터가 여러 개라면 평균으로 통합
df = (
    df.groupby(["연도", "월"], as_index=False)
    ["평균기온(℃)"].mean()
)

min_year = int(df["연도"].min())
max_year = int(df["연도"].max())

# ---------------------------------
# 4. 사용자 입력
# ---------------------------------
st.subheader("📅 분석 조건 선택")

col1, col2 = st.columns(2)

month_names = {
    1: "1월", 2: "2월", 3: "3월",
    4: "4월", 5: "5월", 6: "6월",
    7: "7월", 8: "8월", 9: "9월",
    10: "10월", 11: "11월", 12: "12월"
}

with col1:
    selected_month = st.selectbox(
        "분석할 월을 선택하세요.",
        list(month_names.keys()),
        format_func=lambda x: month_names[x]
    )

with col2:
    prediction_year = st.number_input(
        "예측할 연도를 입력하세요.",
        min_value=max_year + 1,
        max_value=2100,
        value=max_year + 5,
        step=1
    )

# ---------------------------------
# 5. 선택한 월의 데이터 추출
# ---------------------------------
monthly_df = df[
    df["월"] == selected_month
].copy()

monthly_df = monthly_df.sort_values("연도")

if len(monthly_df) < 2:
    st.error("선형회귀 분석을 위한 데이터가 부족합니다.")
    st.stop()

# ---------------------------------
# 6. 선형회귀 직접 구현
# ---------------------------------
# x = 연도
# y = 평균기온

x = monthly_df["연도"].astype(float).tolist()
y = monthly_df["평균기온(℃)"].astype(float).tolist()

n = len(x)

mean_x = sum(x) / n
mean_y = sum(y) / n

numerator = sum(
    (x[i] - mean_x) * (y[i] - mean_y)
    for i in range(n)
)

denominator = sum(
    (x[i] - mean_x) ** 2
    for i in range(n)
)

if denominator == 0:
    st.error("연도 데이터가 충분하지 않아 회귀선을 계산할 수 없습니다.")
    st.stop()

# 기울기
slope = numerator / denominator

# 절편
intercept = mean_y - slope * mean_x

# 예측 함수
def predict(year):
    return slope * year + intercept

predicted_temp = predict(prediction_year)

# ---------------------------------
# 7. 결정계수 R² 계산
# ---------------------------------
y_mean = mean_y

ss_total = sum(
    (value - y_mean) ** 2
    for value in y
)

ss_residual = sum(
    (y[i] - predict(x[i])) ** 2
    for i in range(n)
)

if ss_total == 0:
    r_squared = 1.0 if ss_residual == 0 else 0.0
else:
    r_squared = 1 - ss_residual / ss_total

# ---------------------------------
# 8. 예측 결과 출력
# ---------------------------------
st.divider()
st.subheader("🔮 기온 예측 결과")

result1, result2, result3 = st.columns(3)

with result1:
    st.metric(
        "선택한 월",
        month_names[selected_month]
    )

with result2:
    st.metric(
        "예측 연도",
        f"{int(prediction_year)}년"
    )

with result3:
    st.metric(
        "예측 평균기온",
        f"{predicted_temp:.2f} ℃"
    )

st.info(
    f"📌 {int(prediction_year)}년 "
    f"{month_names[selected_month]}의 예상 평균기온은 "
    f"**{predicted_temp:.2f}℃**입니다."
)

# ---------------------------------
# 9. 그래프 시각화
# ---------------------------------
st.divider()
st.subheader("📈 과거 평균기온과 미래 예측 그래프")

fig, ax = plt.subplots(figsize=(13, 6))

# 실제 기온 데이터
ax.scatter(
    x,
    y,
    color="royalblue",
    s=55,
    label="Actual Temperature",
    zorder=3
)

# 실제 기온을 연결한 선
ax.plot(
    x,
    y,
    color="skyblue",
    linewidth=1.5,
    alpha=0.7,
    label="Historical Temperature"
)

# 회귀선
line_start = min(min(x), prediction_year)
line_end = max(max(x), prediction_year)

line_x = [
    line_start + i * (line_end - line_start) / 200
    for i in range(201)
]

line_y = [predict(year) for year in line_x]

ax.plot(
    line_x,
    line_y,
    color="red",
    linewidth=2.5,
    label="Linear Regression Line"
)

# 예측 기온
ax.scatter(
    [prediction_year],
    [predicted_temp],
    color="green",
    s=180,
    marker="*",
    label="Predicted Temperature",
    zorder=5
)

# 예측값 표시
ax.annotate(
    f"{predicted_temp:.2f} ℃",
    xy=(prediction_year, predicted_temp),
    xytext=(10, 15),
    textcoords="offset points",
    fontsize=11,
    color="green",
    fontweight="bold"
)

# 회귀선과 실제 기온 사이의 차이 표시
for year, actual in zip(x, y):
    fitted = predict(year)

    ax.plot(
        [year, year],
        [actual, fitted],
        color="gray",
        linestyle=":",
        alpha=0.5,
        linewidth=1
    )

# 그래프 설정
ax.set_title(
    f"{month_names[selected_month]} Average Temperature Prediction",
    fontsize=16
)

ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Average Temperature (℃)", fontsize=12)

ax.grid(True, alpha=0.3)
ax.legend()

ax.xaxis.set_major_locator(
    ticker.MaxNLocator(integer=True, nbins=12)
)

fig.tight_layout()

st.pyplot(fig)
plt.close(fig)

# ---------------------------------
# 10. 선형회귀 분석 결과
# ---------------------------------
st.subheader("📊 선형회귀 분석 정보")

info1, info2 = st.columns(2)

with info1:
    st.write(f"**기울기:** {slope:.4f}")
    st.write(f"**절편:** {intercept:.4f}")

with info2:
    st.write(f"**결정계수 R²:** {r_squared:.4f}")
    st.write(f"**분석 데이터 개수:** {n}개")

st.write(
    f"**회귀식:** "
    f"평균기온 = {slope:.4f} × 연도 + ({intercept:.4f})"
)

# ---------------------------------
# 11. 실제 데이터와 예측 데이터 표
# ---------------------------------
st.divider()
st.subheader("📋 연도별 평균기온 데이터")

display_df = monthly_df[
    ["연도", "평균기온(℃)"]
].copy()

display_df["구분"] = "실제 관측값"

prediction_row = pd.DataFrame({
    "연도": [int(prediction_year)],
    "평균기온(℃)": [predicted_temp],
    "구분": ["선형회귀 예측값"]
})

display_df = pd.concat(
    [display_df, prediction_row],
    ignore_index=True
)

display_df["평균기온(℃)"] = (
    display_df["평균기온(℃)"].round(2)
)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)

# ---------------------------------
# 12. 선형회귀 알고리즘 쉽게 설명
# ---------------------------------
st.divider()
st.subheader("💡 선형회귀 알고리즘이란?")

st.markdown("""
### 1️⃣ 선형회귀는 무엇일까요?

선형회귀는 과거에 수집한 데이터를 분석하여
데이터 사이의 관계를 가장 잘 나타내는 직선을 찾는 알고리즘입니다.

이번 프로그램에서는 다음과 같이 사용합니다.

- 가로축(X): 연도
- 세로축(Y): 해당 월의 평균기온

과거 여러 해의 같은 월 기온을 점으로 표시한 뒤,
그 점들과 가장 잘 맞는 직선을 찾아 미래의 기온을 예측합니다.

### 2️⃣ 그래프는 어떻게 읽을까요?

- 🔵 파란색 점: 실제로 관측된 과거 평균기온
- 🔵 연한 파란색 선: 과거 기온의 변화
- 🔴 빨간색 직선: 선형회귀가 찾아낸 추세선
- 🟢 초록색 별: 선택한 연도의 예측 평균기온
- ⚪ 회색 점선: 실제 기온과 회귀선의 차이

### 3️⃣ 알고리즘은 어떻게 작동할까요?

① 과거의 연도와 평균기온 데이터를 모읍니다.

② 데이터의 전체적인 흐름을 분석합니다.

③ 실제 데이터와 직선 사이의 차이가 작아지도록
직선의 기울기와 절편을 계산합니다.

④ 계산된 직선에 미래의 연도를 넣어
예상 평균기온을 계산합니다.

### 4️⃣ 기울기는 무엇을 의미할까요?

- 기울기가 양수(+)라면 장기적으로 기온이 상승하는 경향입니다.
- 기울기가 음수(-)라면 장기적으로 기온이 하락하는 경향입니다.
- 기울기가 0에 가까우면 장기적인 변화가 작다는 의미입니다.

### 5️⃣ 결정계수 R²는 무엇일까요?

결정계수는 회귀선이 과거 데이터의 변화를
얼마나 잘 설명하는지 나타내는 값입니다.

- 1에 가까울수록 과거 데이터와 회귀선이 잘 맞습니다.
- 0에 가까울수록 회귀선이 과거 기온의 변화를 잘 설명하지 못합니다.

단, R²가 높더라도 미래 예측이 반드시 정확하다는 뜻은 아닙니다.
""")

# ---------------------------------
# 13. 예측 시 주의사항
# ---------------------------------
st.warning("""
⚠️ **예측 결과를 해석할 때 주의하세요!**

이 프로그램은 과거의 같은 월 평균기온에 나타난
장기적인 직선 추세를 바탕으로 미래 기온을 예측합니다.

계절 변화, 기후변화, 이상기후 등 여러 요인을
모두 반영하는 기상예보 모델은 아닙니다.

따라서 예측 결과는 실제 미래 기온과 다를 수 있습니다.
""")

st.divider()

st.caption(
    "🌡️ 월별 평균기온 분석 및 선형회귀 예측 프로그램"
)
