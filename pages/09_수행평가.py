
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# ==========================================
# 1. 페이지 설정
# ==========================================

st.set_page_config(
    page_title="SRT 승차인원 분석",
    page_icon="🚄",
    layout="wide"
)

st.title("🚄 SRT 주간 승차인원 분석")
st.write(
    "주차별 승차인원을 분석하고 "
    "인공지능 알고리즘으로 미래 승차인원을 예측합니다."
)

st.divider()


# ==========================================
# 2. CSV 불러오기
# ==========================================

@st.cache_data
def load_data(file_bytes=None):

    import io

    if file_bytes is not None:
        for encoding in ["cp949", "utf-8-sig", "utf-8"]:
            try:
                return pd.read_csv(
                    io.BytesIO(file_bytes),
                    encoding=encoding
                )
            except UnicodeDecodeError:
                continue

        raise ValueError("CSV 인코딩을 확인해 주세요.")

    base_dir = Path(__file__).resolve().parent.parent

    file_path = (
        base_dir /
        "(주)에스알_주별 평균 승차인원_20250930.csv"
    )

    if not file_path.exists():
        return None

    for encoding in ["cp949", "utf-8-sig", "utf-8"]:
        try:
            return pd.read_csv(
                file_path,
                encoding=encoding
            )
        except UnicodeDecodeError:
            continue

    raise ValueError("CSV 파일 인코딩 오류")


uploaded_file = st.file_uploader(
    "CSV 파일 업로드",
    type=["csv"]
)

try:
    if uploaded_file:
        df = load_data(uploaded_file.getvalue())
    else:
        df = load_data()

except Exception as e:
    st.error(f"파일을 읽을 수 없습니다: {e}")
    st.stop()


if df is None:
    st.error(
        "CSV 파일을 찾을 수 없습니다. "
        "pages 폴더의 상위 폴더에 CSV를 넣어 주세요."
    )
    st.stop()


# ==========================================
# 3. 데이터 전처리
# ==========================================

df.columns = df.columns.str.strip()

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

df = df.dropna(subset=station_columns, how="all")

df[station_columns] = df[station_columns].fillna(0)

df = (
    df.groupby("주차", as_index=False)[station_columns]
    .sum()
    .sort_values("주차")
)

df["전체승차인원"] = df[station_columns].sum(axis=1)

if len(df) < 4:
    st.error("분석하려면 최소 4주 이상의 데이터가 필요합니다.")
    st.stop()

last_week = int(df["주차"].max())

st.success(
    f"데이터 분석 준비 완료! "
    f"{len(df)}개 주차 / {len(station_columns)}개 역"
)


# ==========================================
# 4. 주차 범위 선택
# ==========================================

st.header("📅 분석할 주차 선택")

available_weeks = sorted(df["주차"].unique())

col1, col2 = st.columns(2)

with col1:
    start_week = st.selectbox(
        "시작 주차",
        available_weeks,
        index=0,
        format_func=lambda x: f"{x}주차"
    )

with col2:
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
    f"선택한 분석 기간: {start_week}주차 ~ {end_week}주차"
)


# ==========================================
# 5. 역별 승차인원 TOP 10
# ==========================================

st.divider()
st.header("🏆 가장 많이 이용한 역")

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

c1, c2, c3 = st.columns(3)

c1.metric("🥇 승차인원 1위 역", top_station)
c2.metric("🚄 1위 역 승차인원", f"{top_passengers:,}명")
c3.metric("👥 전체 승차인원", f"{total_passengers:,}명")


top10 = station_total.head(10).sort_values()

fig_station = go.Figure()

fig_station.add_trace(
    go.Bar(
        x=top10.values,
        y=top10.index,
        orientation="h",
        text=[
            f"{int(v):,}명" for v in top10.values
        ],
        textposition="outside"
    )
)

fig_station.update_layout(
    title="역별 승차인원 TOP 10",
    xaxis_title="승차인원",
    yaxis_title="역",
    template="plotly_white",
    height=500
)

st.plotly_chart(
    fig_station,
    use_container_width=True
)


# ==========================================
# 6. 가장 많이 탄 주차
# ==========================================

st.divider()
st.header("📈 가장 많이 이용한 주차")

max_row = selected_df.loc[
    selected_df["전체승차인원"].idxmax()
]

max_week = int(max_row["주차"])
max_week_passengers = int(
    max_row["전체승차인원"]
)

c1, c2 = st.columns(2)

c1.metric("🏆 최다 승차 주차", f"{max_week}주차")
c2.metric(
    "승차인원",
    f"{max_week_passengers:,}명"
)

fig_week = go.Figure()

fig_week.add_trace(
    go.Bar(
        x=selected_df["주차"],
        y=selected_df["전체승차인원"],
        name="실제 승차인원"
    )
)

fig_week.add_trace(
    go.Scatter(
        x=[max_week],
        y=[max_week_passengers],
        mode="markers+text",
        name="최다 승차 주차",
        marker=dict(
            color="red",
            size=15,
            symbol="star"
        ),
        text=["최다"],
        textposition="top center"
    )
)

fig_week.update_layout(
    title="주차별 전체 승차인원",
    xaxis_title="주차",
    yaxis_title="승차인원",
    xaxis=dict(dtick=1),
    template="plotly_white",
    height=450
)

st.plotly_chart(
    fig_week,
    use_container_width=True
)


# ==========================================
# 7. 미래 예측 설정
# ==========================================

st.divider()
st.header("🔮 미래 승차인원 예측")

st.write(
    f"현재 데이터의 마지막 주차는 {last_week}주차입니다. "
    "이후 주차의 승차인원을 예측합니다."
)

forecast_count = st.slider(
    "몇 주 후까지 예측할까요?",
    min_value=1,
    max_value=12,
    value=4
)

exclude_last = st.checkbox(
    "마지막 주차 데이터는 부분 집계일 수 있으므로 "
    "예측 학습에서 제외하기",
    value=True
)

# 예측 학습 데이터
train_df = df.copy()

if exclude_last and len(train_df) > 4:
    train_df = train_df[
        train_df["주차"] < last_week
    ].copy()

if len(train_df) < 4:
    st.warning("예측 학습 데이터가 부족합니다.")
    st.stop()

X = train_df[["주차"]].values

y = train_df["전체승차인원"].values

future_weeks = np.arange(
    last_week + 1,
    last_week + forecast_count + 1
)

future_X = future_weeks.reshape(-1, 1)


# ==========================================
# 8. 알고리즘 3가지
# ==========================================

models = {
    "선형회귀": LinearRegression(),

    "랜덤 포레스트": RandomForestRegressor(
        n_estimators=200,
        max_depth=4,
        random_state=42
    ),

    "의사결정나무": DecisionTreeRegressor(
        max_depth=4,
        random_state=42
    )
}

predictions = {}
results = []

for name, model in models.items():

    # 모델 학습
    model.fit(X, y)

    # 과거 학습 데이터 예측
    train_prediction = model.predict(X)

    # 미래 예측
    future_prediction = model.predict(future_X)

    # 승차인원이 음수가 될 수 없도록 처리
    future_prediction = np.maximum(
        future_prediction,
        0
    )

    predictions[name] = future_prediction

    mae = mean_absolute_error(
        y,
        train_prediction
    )

    r2 = r2_score(
        y,
        train_prediction
    )

    results.append({
        "알고리즘": name,
        "평균절대오차(MAE)": round(mae, 2),
        "결정계수(R²)": round(r2, 3)
    })


# ==========================================
# 9. 미래 예측 그래프
# ==========================================

st.subheader("📊 실제 승차인원과 미래 예측 비교")

fig_future = go.Figure()

# 실제 데이터
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

colors = {
    "선형회귀": "red",
    "랜덤 포레스트": "blue",
    "의사결정나무": "green"
}

# 알고리즘별 예측선
for name, prediction in predictions.items():

    # 마지막 실제 데이터와 미래 예측 연결
    graph_x = np.concatenate(
        ([last_week], future_weeks)
    )

    last_actual = float(
        df.loc[
            df["주차"] == last_week,
            "전체승차인원"
        ].iloc[0]
    )

    graph_y = np.concatenate(
        ([last_actual], prediction)
    )

    fig_future.add_trace(
        go.Scatter(
            x=graph_x,
            y=graph_y,
            mode="lines+markers",
            name=name,
            line=dict(
                color=colors[name],
                width=3,
                dash="dash"
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
    title="SRT 미래 주차별 승차인원 예측",
    xaxis_title="주차",
    yaxis_title="승차인원 (명)",
    xaxis=dict(
        dtick=1
    ),
    template="plotly_white",
    height=550,
    hovermode="x unified",
    legend_title="분석 구분"
)

st.plotly_chart(
    fig_future,
    use_container_width=True
)


# ==========================================
# 10. 미래 예측값 비교표
# ==========================================

st.subheader("🔢 미래 주차별 예측 승차인원")

forecast_table = pd.DataFrame({
    "예측 주차": [
        f"{w}주차" for w in future_weeks
    ],
    "선형회귀": predictions["선형회귀"],
    "랜덤 포레스트": predictions["랜덤 포레스트"],
    "의사결정나무": predictions["의사결정나무"]
})

for col in [
    "선형회귀",
    "랜덤 포레스트",
    "의사결정나무"
]:
    forecast_table[col] = (
        forecast_table[col].round(0).astype(int)
    )

st.dataframe(
    forecast_table.style.format({
        "선형회귀": "{:,}명",
        "랜덤 포레스트": "{:,}명",
        "의사결정나무": "{:,}명"
    }),
    use_container_width=True,
    hide_index=True
)


# ==========================================
# 11. 미래 승차인원 요약
# ==========================================

st.subheader("🏆 알고리즘별 미래 예측 요약")

summary = []

for name, prediction in predictions.items():

    highest_index = int(
        np.argmax(prediction)
    )

    summary.append({
        "알고리즘": name,
        "가장 많은 승차인원이 예상되는 주차":
            f"{future_weeks[highest_index]}주차",
        "최대 예측 승차인원":
            f"{int(round(prediction[highest_index])):,}명"
    })

st.dataframe(
    pd.DataFrame(summary),
    use_container_width=True,
    hide_index=True
)


# ==========================================
# 12. 알고리즘 성능 비교
# ==========================================

st.divider()
st.header("🤖 알고리즘 성능 비교")

result_df = pd.DataFrame(results)

st.dataframe(
    result_df,
    use_container_width=True,
    hide_index=True
)

st.caption(
    "MAE는 학습 데이터에서 예측값과 실제값의 평균 차이입니다. "
    "R²는 학습 데이터에 대한 설명력을 나타냅니다. "
    "이 값은 학습 데이터 기준이므로 미래 예측의 정확도를 "
    "직접 보장하지 않습니다."
)


# ==========================================
# 13. 알고리즘 설명
# ==========================================

st.divider()
st.header("📚 알고리즘 알아보기")

with st.expander("① 선형회귀"):

    st.write(
        "과거 주차와 승차인원 사이의 직선 관계를 학습합니다. "
        "전체적인 증가 또는 감소 추세를 미래로 연장합니다."
    )

    st.latex("y = ax + b")

with st.expander("② 랜덤 포레스트"):

    st.write(
        "여러 개의 의사결정나무를 학습시키고 "
        "그 결과를 평균하여 예측합니다. "
        "복잡한 패턴을 학습할 수 있지만 "
        "학습 범위 밖의 미래 추세를 연장하는 데는 한계가 있습니다."
    )

with st.expander("③ 의사결정나무"):

    st.write(
        "주차에 따라 데이터를 여러 구간으로 나누어 "
        "각 구간의 승차인원 패턴을 학습합니다. "
        "미래 주차가 학습 범위를 벗어나면 "
        "마지막 구간의 예측값이 유지될 수 있습니다."
    )


# ==========================================
# 14. 최종 분석 결과
# ==========================================

st.divider()
st.header("📝 전체 분석 결과")

st.markdown(
    f"""
    ### 📊 과거 데이터 분석

    - 분석 주차: {start_week}주차 ~ {end_week}주차
    - 가장 많이 탄 역: **{top_station}**
    - 해당 역 승차인원: **{top_passengers:,}명**
    - 가장 많이 탄 주차: **{max_week}주차**
    - 해당 주차 전체 승차인원: **{max_week_passengers:,}명**

    ### 🔮 미래 예측

    - 마지막 데이터 주차: **{last_week}주차**
    - 예측 기간: **{last_week + 1}주차 ~ {last_week + forecast_count}주차**
    - 사용 알고리즘: **선형회귀, 랜덤 포레스트, 의사결정나무**
    """
)

st.success("SRT 승차인원 분석 및 미래 예측 완료! 🚄")

st.caption(
    "주의: 미래 예측은 과거 패턴에 기반한 추정치입니다. "
    "실제 승차인원은 명절, 휴가철, 요일, 날씨, 열차 운행 변화 등에 "
    "따라 달라질 수 있습니다."
)
