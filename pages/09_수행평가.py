
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from pathlib import Path
from io import BytesIO

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# 1. 페이지 설정
# ==========================================

st.set_page_config(
    page_title="SRT 종합 분석 시스템",
    page_icon="🚄",
    layout="wide"
)

st.title("🚄 SRT 종합 승차인원 분석 시스템")
st.markdown(
    "주차별 승차인원 분석부터 인공지능 미래 예측까지!"
)

st.divider()


# ==========================================
# 2. CSV 파일 불러오기
# ==========================================

@st.cache_data
def load_data(file_bytes=None):

    base_dir = Path(__file__).resolve().parent.parent

    if file_bytes is not None:
        source = BytesIO(file_bytes)
    else:
        source = (
            base_dir /
            "(주)에스알_주별 평균 승차인원_20250930.csv"
        )

        if not source.exists():
            return None

    for encoding in ["cp949", "utf-8-sig", "utf-8"]:
        try:
            if file_bytes is not None:
                source.seek(0)

            return pd.read_csv(
                source,
                encoding=encoding
            )

        except UnicodeDecodeError:
            continue

    raise ValueError("CSV 인코딩을 확인해 주세요.")


uploaded_file = st.file_uploader(
    "📂 SRT CSV 파일 업로드",
    type=["csv"]
)

try:
    if uploaded_file is not None:
        df = load_data(uploaded_file.getvalue())
    else:
        df = load_data()

except Exception as e:
    st.error(f"파일 오류: {e}")
    st.stop()

if df is None:
    st.error(
        "CSV 파일을 찾을 수 없습니다. "
        "pages 폴더의 상위 폴더에 파일을 넣어 주세요."
    )
    st.stop()


# ==========================================
# 3. 데이터 전처리
# ==========================================

df.columns = df.columns.astype(str).str.strip()

if "주차구분" not in df.columns:
    st.error("주차구분 컬럼이 없습니다.")
    st.stop()

df["주차"] = pd.to_numeric(
    df["주차구분"]
    .astype(str)
    .str.extract(r"(\d+)\s*주차")[0],
    errors="coerce"
)

df = df.dropna(subset=["주차"]).copy()
df["주차"] = df["주차"].astype(int)

station_columns = [
    col for col in df.columns
    if col not in ["주차구분", "주차"]
]

for col in station_columns:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

df = df.dropna(
    subset=station_columns,
    how="all"
)

df[station_columns] = (
    df[station_columns].fillna(0)
)

# 같은 주차가 여러 행이면 합산
df = (
    df.groupby("주차", as_index=False)[station_columns]
    .sum()
    .sort_values("주차")
)

df["전체승차인원"] = df[station_columns].sum(axis=1)

if len(df) < 8:
    st.error("분석을 위해 최소 8주 이상의 데이터가 필요합니다.")
    st.stop()

last_week = int(df["주차"].max())

st.success(
    f"데이터 분석 준비 완료! "
    f"{len(df)}개 주차 / {len(station_columns)}개 역"
)


# ==========================================
# 4. 주차 범위 선택
# ==========================================

st.header("📅 분석 기간 선택")

available_weeks = sorted(df["주차"].unique())

c1, c2 = st.columns(2)

with c1:
    start_week = st.selectbox(
        "시작 주차",
        available_weeks,
        index=0,
        format_func=lambda x: f"{x}주차"
    )

with c2:
    end_options = [
        w for w in available_weeks
        if w >= start_week
    ]

    end_week = st.selectbox(
        "마지막 주차",
        end_options,
        index=len(end_options) - 1,
        format_func=lambda x: f"{x}주차"
    )

selected_df = df[
    (df["주차"] >= start_week) &
    (df["주차"] <= end_week)
].copy()

st.info(
    f"분석 기간: {start_week}주차 ~ {end_week}주차"
)


# ==========================================
# 5. 종합 현황
# ==========================================

st.divider()
st.header("🏆 1. 종합 현황")

station_total = (
    selected_df[station_columns]
    .sum()
    .sort_values(ascending=False)
)

top_station = station_total.index[0]
top_passengers = int(station_total.iloc[0])

total_passengers = int(
    selected_df["전체승차인원"].sum()
)

max_row = selected_df.loc[
    selected_df["전체승차인원"].idxmax()
]

max_week = int(max_row["주차"])
max_week_passengers = int(
    max_row["전체승차인원"]
)

c1, c2, c3, c4 = st.columns(4)

c1.metric("🥇 1위 역", top_station)
c2.metric("🚄 1위 역 승차인원", f"{top_passengers:,}명")
c3.metric("🏆 최다 이용 주차", f"{max_week}주차")
c4.metric("👥 전체 승차인원", f"{total_passengers:,}명")


# ==========================================
# 6. 역별 TOP 10
# ==========================================

st.divider()
st.header("📊 2. 역별 승차인원 순위")

top10 = station_total.head(10).sort_values()

fig = go.Figure()

fig.add_trace(
    go.Bar(
        x=top10.values,
        y=top10.index,
        orientation="h",
        text=[
            f"{int(v):,}명"
            for v in top10.values
        ],
        textposition="outside"
    )
)

fig.update_layout(
    title="역별 승차인원 TOP 10",
    xaxis_title="승차인원",
    yaxis_title="역",
    template="plotly_white",
    height=500
)

st.plotly_chart(fig, use_container_width=True)


# ==========================================
# 7. 주차별 승차인원
# ==========================================

st.divider()
st.header("📈 3. 주차별 승차인원")

fig = go.Figure()

fig.add_trace(
    go.Bar(
        x=selected_df["주차"],
        y=selected_df["전체승차인원"],
        name="전체 승차인원"
    )
)

fig.add_trace(
    go.Scatter(
        x=[max_week],
        y=[max_week_passengers],
        mode="markers+text",
        marker=dict(
            color="red",
            size=16,
            symbol="star"
        ),
        text=["최다 이용"],
        textposition="top center",
        name="최다 이용 주차"
    )
)

fig.update_layout(
    title="주차별 전체 승차인원",
    xaxis_title="주차",
    yaxis_title="승차인원",
    xaxis=dict(dtick=1),
    template="plotly_white",
    height=450
)

st.plotly_chart(fig, use_container_width=True)


# ==========================================
# 8. 추가기능 ① 역별 승차인원 히트맵
# ==========================================

st.divider()
st.header("🌈 4. 역별 승차인원 히트맵")

st.write(
    "색이 진할수록 해당 역의 승차인원이 많습니다."
)

heat_df = selected_df.set_index("주차")[station_columns].T

fig_heat = px.imshow(
    heat_df,
    labels={
        "x": "주차",
        "y": "역 이름",
        "color": "승차인원"
    },
    aspect="auto",
    color_continuous_scale="YlOrRd",
    title="역별 · 주차별 승차인원 분포"
)

fig_heat.update_layout(height=750)

st.plotly_chart(
    fig_heat,
    use_container_width=True
)


# ==========================================
# 9. 추가기능 ② 이상치 자동 탐지
# ==========================================

st.divider()
st.header("🚨 5. 이상치 자동 탐지")

st.write(
    "주차별 전체 승차인원을 IQR 방식으로 분석합니다. "
    "평소 범위에서 크게 벗어난 주차를 찾아냅니다."
)

values = selected_df["전체승차인원"]

q1 = values.quantile(0.25)
q3 = values.quantile(0.75)
iqr = q3 - q1

lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

anomaly_df = selected_df[
    (values < lower) |
    (values > upper)
].copy()

fig_anomaly = go.Figure()

fig_anomaly.add_trace(
    go.Scatter(
        x=selected_df["주차"],
        y=selected_df["전체승차인원"],
        mode="lines+markers",
        name="실제 승차인원"
    )
)

fig_anomaly.add_hrect(
    y0=lower,
    y1=upper,
    fillcolor="green",
    opacity=0.08,
    line_width=0,
    annotation_text="일반적인 데이터 범위"
)

if not anomaly_df.empty:
    fig_anomaly.add_trace(
        go.Scatter(
            x=anomaly_df["주차"],
            y=anomaly_df["전체승차인원"],
            mode="markers+text",
            marker=dict(
                color="red",
                size=14,
                symbol="x"
            ),
            text=[
                f"{w}주차"
                for w in anomaly_df["주차"]
            ],
            textposition="top center",
            name="이상치"
        )
    )

fig_anomaly.update_layout(
    title="주차별 이상치 탐지",
    xaxis_title="주차",
    yaxis_title="승차인원",
    template="plotly_white",
    height=450
)

st.plotly_chart(
    fig_anomaly,
    use_container_width=True
)

if anomaly_df.empty:
    st.success("선택한 기간에 이상치가 발견되지 않았습니다.")
else:
    st.warning(
        f"총 {len(anomaly_df)}개의 이상치가 발견되었습니다."
    )

    st.dataframe(
        anomaly_df[
            ["주차", "전체승차인원"]
        ].assign(
            주차=lambda x: x["주차"].astype(str) + "주차"
        ),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "이상치는 통계적으로 평소 범위에서 벗어난 값입니다. "
        "데이터 오류나 실제 이용객 급변을 뜻한다고 단정할 수 없습니다."
    )


# ==========================================
# 10. 추가기능 ③ 알고리즘 정확도 대결
# ==========================================

st.divider()
st.header("🤖 6. 인공지능 알고리즘 정확도 대결")

st.write(
    "과거 데이터를 시간 순서대로 학습용과 검증용으로 나누어 "
    "세 가지 알고리즘의 예측 성능을 비교합니다."
)

model_factories = {
    "선형회귀": lambda: LinearRegression(),

    "랜덤 포레스트": lambda: RandomForestRegressor(
        n_estimators=200,
        max_depth=4,
        random_state=42
    ),

    "의사결정나무": lambda: DecisionTreeRegressor(
        max_depth=4,
        random_state=42
    )
}

# 시간 순서 유지: 앞쪽 학습, 뒤쪽 검증
validation_df = df.sort_values("주차").copy()

split_index = int(len(validation_df) * 0.8)

train_df = validation_df.iloc[:split_index]
test_df = validation_df.iloc[split_index:]

X_train = train_df[["주차"]].values
y_train = train_df["전체승차인원"].values

X_test = test_df[["주차"]].values
y_test = test_df["전체승차인원"].values

validation_results = []
validation_predictions = {}

for name, factory in model_factories.items():

    model = factory()

    model.fit(X_train, y_train)

    pred = np.maximum(
        model.predict(X_test),
        0
    )

    validation_predictions[name] = pred

    mae = mean_absolute_error(y_test, pred)

    rmse = np.sqrt(
        mean_squared_error(y_test, pred)
    )

    r2 = r2_score(y_test, pred)

    validation_results.append({
        "알고리즘": name,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R²": round(r2, 3)
    })

validation_result_df = pd.DataFrame(
    validation_results
)

# MAE가 가장 낮은 알고리즘
best_model = validation_result_df.loc[
    validation_result_df["MAE"].idxmin(),
    "알고리즘"
]

st.metric(
    "🏆 검증 데이터 기준 MAE 1위",
    best_model
)

st.caption(
    "MAE와 RMSE는 낮을수록 좋고, R²는 높을수록 좋습니다. "
    "검증 데이터는 과거 데이터의 마지막 20%입니다."
)

st.dataframe(
    validation_result_df,
    use_container_width=True,
    hide_index=True
)

# 검증 실제값과 예측값 그래프
fig_validation = go.Figure()

fig_validation.add_trace(
    go.Scatter(
        x=test_df["주차"],
        y=y_test,
        mode="lines+markers",
        name="실제 승차인원",
        line=dict(color="black", width=3)
    )
)

colors = {
    "선형회귀": "red",
    "랜덤 포레스트": "blue",
    "의사결정나무": "green"
}

for name, pred in validation_predictions.items():

    fig_validation.add_trace(
        go.Scatter(
            x=test_df["주차"],
            y=pred,
            mode="lines+markers",
            name=name,
            line=dict(
                color=colors[name],
                dash="dash"
            )
        )
    )

fig_validation.update_layout(
    title="검증 데이터 실제값과 알고리즘 예측값",
    xaxis_title="주차",
    yaxis_title="승차인원",
    template="plotly_white",
    height=450
)

st.plotly_chart(
    fig_validation,
    use_container_width=True
)


# ==========================================
# 11. 미래 승차인원 예측
# ==========================================

st.divider()
st.header("🔮 7. 미래 승차인원 예측")

forecast_count = st.slider(
    "미래 몇 주까지 예측할까요?",
    min_value=1,
    max_value=12,
    value=4
)

exclude_last = st.checkbox(
    "마지막 주차를 학습에서 제외하기 "
    "(부분 집계 가능성 확인)",
    value=True
)

forecast_train = df.copy()

if exclude_last and len(forecast_train) > 4:
    forecast_train = forecast_train[
        forecast_train["주차"] < last_week
    ]

future_weeks = np.arange(
    last_week + 1,
    last_week + forecast_count + 1
)

future_X = future_weeks.reshape(-1, 1)

X_all = forecast_train[["주차"]].values
y_all = forecast_train["전체승차인원"].values

future_predictions = {}

for name, factory in model_factories.items():

    model = factory()

    model.fit(X_all, y_all)

    future_predictions[name] = np.maximum(
        model.predict(future_X),
        0
    )


# 미래 예측 그래프
fig_future = go.Figure()

fig_future.add_trace(
    go.Scatter(
        x=df["주차"],
        y=df["전체승차인원"],
        mode="lines+markers",
        name="실제 승차인원",
        line=dict(
            color="black",
            width=3
        )
    )
)

for name, pred in future_predictions.items():

    last_actual = float(
        df["전체승차인원"].iloc[-1]
    )

    graph_x = np.concatenate(
        ([last_week], future_weeks)
    )

    graph_y = np.concatenate(
        ([last_actual], pred)
    )

    fig_future.add_trace(
        go.Scatter(
            x=graph_x,
            y=graph_y,
            mode="lines+markers",
            name=name,
            line=dict(
                color=colors[name],
                dash="dash",
                width=3
            )
        )
    )

fig_future.add_vline(
    x=last_week,
    line_dash="dot",
    line_color="gray",
    annotation_text="미래 예측 시작"
)

fig_future.update_layout(
    title="SRT 미래 승차인원 예측",
    xaxis_title="주차",
    yaxis_title="승차인원",
    xaxis=dict(dtick=1),
    template="plotly_white",
    height=550,
    hovermode="x unified"
)

st.plotly_chart(
    fig_future,
    use_container_width=True
)

forecast_table = pd.DataFrame({
    "예측 주차": [
        f"{w}주차" for w in future_weeks
    ],
    "선형회귀": future_predictions["선형회귀"].round(0).astype(int),
    "랜덤 포레스트": future_predictions["랜덤 포레스트"].round(0).astype(int),
    "의사결정나무": future_predictions["의사결정나무"].round(0).astype(int)
})

st.subheader("미래 승차인원 예측값")

st.dataframe(
    forecast_table,
    use_container_width=True,
    hide_index=True
)


# ==========================================
# 12. 추가기능 ④ 자동 분석 보고서
# ==========================================

st.divider()
st.header("📝 8. 자동 데이터 분석 보고서")

# 전주 대비 변화
weekly = selected_df.sort_values("주차").copy()

if len(weekly) >= 2:

    first_value = float(
        weekly["전체승차인원"].iloc[0]
    )

    last_value = float(
        weekly["전체승차인원"].iloc[-1]
    )

    if first_value != 0:
        change_rate = (
            (last_value - first_value) /
            first_value * 100
        )
    else:
        change_rate = 0

else:
    change_rate = 0

# 역별 점유율
if total_passengers > 0:
    top_share = (
        top_passengers /
        total_passengers * 100
    )
else:
    top_share = 0

# 알고리즘별 미래 예측 평균
future_means = {
    name: float(np.mean(pred))
    for name, pred in future_predictions.items()
}

report = f"""
# 🚄 SRT 승차인원 분석 보고서

## 1. 분석 개요

- 분석 기간: {start_week}주차 ~ {end_week}주차
- 분석 주차 수: {len(selected_df)}주
- 분석 역 개수: {len(station_columns)}개

## 2. 역별 분석

- 가장 많이 이용한 역: {top_station}
- 해당 역 승차인원 합계: {top_passengers:,}명
- 전체 승차인원 중 해당 역 비율: {top_share:.1f}%

## 3. 주차별 분석

- 승차인원이 가장 많은 주차: {max_week}주차
- 해당 주차 승차인원: {max_week_passengers:,}명
- 선택 기간 첫 주와 마지막 주의 변화율: {change_rate:+.1f}%

## 4. 이상치 분석

- 발견된 이상치: {len(anomaly_df)}개

## 5. 알고리즘 검증 결과

- 검증 데이터 기준 MAE가 가장 낮은 알고리즘: {best_model}

## 6. 미래 예측

- 예측 기간: {last_week + 1}주차 ~ {last_week + forecast_count}주차
- 선형회귀 평균 예측: {future_means['선형회귀']:,.0f}명
- 랜덤 포레스트 평균 예측: {future_means['랜덤 포레스트']:,.0f}명
- 의사결정나무 평균 예측: {future_means['의사결정나무']:,.0f}명

## 7. 종합 의견

{top_station}역이 선택한 기간에서 가장 높은 승차인원을 기록했습니다.
{max_week}주차의 전체 승차인원이 분석 기간 중 가장 많았습니다.
세 가지 알고리즘의 검증 결과와 미래 예측값을 비교하여
향후 승차인원 추세를 참고할 수 있습니다.

※ 본 보고서는 통계 분석을 기반으로 자동 생성한 보고서입니다.
미래 예측값은 실제 이용객 수를 보장하지 않습니다.
"""

st.markdown(report)

st.download_button(
    label="📥 분석 보고서 다운로드 (TXT)",
    data=report.encode("utf-8-sig"),
    file_name="SRT_승차인원_분석보고서.txt",
    mime="text/plain"
)


# ==========================================
# 13. 분석 데이터 다운로드
# ==========================================

st.divider()
st.header("💾 9. 분석 결과 다운로드")

csv_data = forecast_table.to_csv(
    index=False,
    encoding="utf-8-sig"
)

st.download_button(
    label="📊 미래 예측 결과 CSV 다운로드",
    data=csv_data,
    file_name="SRT_미래승차인원_예측.csv",
    mime="text/csv"
)

excel_data = BytesIO()

with pd.ExcelWriter(
    excel_data,
    engine="openpyxl"
) as writer:

    selected_df.to_excel(
        writer,
        sheet_name="주차별 분석",
        index=False
    )

    station_total.rename(
        "승차인원 합계"
    ).to_excel(
        writer,
        sheet_name="역별 순위"
    )

    validation_result_df.to_excel(
        writer,
        sheet_name="알고리즘 검증",
        index=False
    )

    forecast_table.to_excel(
        writer,
        sheet_name="미래 예측",
        index=False
    )

    anomaly_df.to_excel(
        writer,
        sheet_name="이상치",
        index=False
    )

st.download_button(
    label="📗 전체 분석 결과 엑셀 다운로드",
    data=excel_data.getvalue(),
    file_name="SRT_종합분석.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)


# ==========================================
# 14. 풍선 축하 효과
# ==========================================

st.divider()

st.markdown(
    """
    <div style="
        text-align:center;
        padding:20px;
        background:linear-gradient(90deg,#eaf5ff,#fceeff);
        border-radius:15px;
    ">
        <h2>🎉 SRT 데이터 분석 완료!</h2>
        <p>데이터 분석부터 미래 예측까지 성공적으로 마쳤습니다.</p>
        <h3>🚄 수고하셨습니다!</h3>
    </div>
    """,
    unsafe_allow_html=True
)

# 분석 조건이 바뀌면 풍선 효과 실행
celebration_key = (
    int(start_week),
    int(end_week),
    int(forecast_count)
)

if st.session_state.get("celebration_key") != celebration_key:
    st.balloons()
    st.session_state["celebration_key"] = celebration_key

# 다시 풍선 터뜨리기
if st.button("🎈 풍선 다시 터뜨리기!"):
    st.balloons()

st.caption(
    "SRT 종합 승차인원 분석 시스템 | "
    "Pandas · Plotly · Scikit-learn"
)
