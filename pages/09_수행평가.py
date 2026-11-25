import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path
import re

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# ==========================================
# 1. 페이지 기본 설정
# ==========================================

st.set_page_config(
    page_title="SRT 주차별 승차인원 분석",
    page_icon="🚄",
    layout="wide"
)

st.title("🚄 SRT 주차별 평균 승차인원 분석")
st.write(
    "주차를 선택하면 역별 승차인원과 주차별 이용 현황을 분석하고, "
    "AI 알고리즘 3가지로 미래 승차인원을 예측합니다."
)


# ==========================================
# 2. CSV 파일 경로
# ==========================================

# 이 파이썬 파일은 pages 폴더 안에 있고,
# CSV 파일은 pages 폴더의 상위 폴더에 있습니다.

CSV_PATH = (
    Path(__file__).resolve().parent.parent
    / "(주)에스알_주별 평균 승차인원_20250930.csv"
)


# ==========================================
# 3. CSV 파일 읽기
# ==========================================

@st.cache_data
def load_data(file_path):

    encodings = [
        "cp949",
        "utf-8-sig",
        "utf-8"
    ]

    last_error = None

    for encoding in encodings:
        try:
            data = pd.read_csv(
                file_path,
                encoding=encoding
            )
            return data

        except Exception as error:
            last_error = error

    raise ValueError(
        f"CSV 파일을 읽을 수 없습니다: {last_error}"
    )


if not CSV_PATH.exists():

    st.error(
        "CSV 파일을 찾을 수 없습니다."
    )

    st.info(
        "CSV 파일을 pages 폴더가 아닌 "
        "프로젝트 최상위 폴더에 넣어주세요."
    )

    st.code(str(CSV_PATH))
    st.stop()


try:

    df = load_data(CSV_PATH)

except Exception as error:

    st.error(
        f"CSV 파일을 불러오는 중 오류가 발생했습니다: {error}"
    )

    st.stop()


# ==========================================
# 4. 데이터 정리
# ==========================================

df.columns = (
    df.columns
    .astype(str)
    .str.replace("\ufeff", "", regex=False)
    .str.strip()
)

if "주차구분" not in df.columns:

    st.error(
        "CSV 파일에서 '주차구분' 컬럼을 찾을 수 없습니다."
    )

    st.write("현재 CSV 컬럼 목록")
    st.write(df.columns.tolist())

    st.stop()


# 주차구분에서 주차 숫자 추출
# 예: 1주차, 2025년 1주차 등

def extract_week(value):

    match = re.search(
        r"(\d+)\s*주차",
        str(value)
    )

    if match:
        return int(match.group(1))

    # '1주'처럼 주차라는 글자가 없는 경우도 처리
    match = re.search(
        r"(\d+)\s*주",
        str(value)
    )

    if match:
        return int(match.group(1))

    return np.nan


df["주차"] = df["주차구분"].apply(
    extract_week
)

df = df.dropna(
    subset=["주차"]
).copy()

df["주차"] = df["주차"].astype(int)


# 역별 숫자 데이터 변환

station_columns = [
    column
    for column in df.columns
    if column not in ["주차구분", "주차"]
]

for column in station_columns:

    df[column] = pd.to_numeric(
        df[column]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip(),
        errors="coerce"
    )


# 역별 데이터가 전부 비어 있는 컬럼 제거

station_columns = [
    column
    for column in station_columns
    if df[column].notna().any()
]

if not station_columns:

    st.error(
        "역별 승차인원 숫자 데이터를 찾을 수 없습니다."
    )

    st.stop()


# 중복 주차가 있다면 합산

df = (
    df.groupby("주차", as_index=False)[station_columns]
    .sum()
    .sort_values("주차")
    .reset_index(drop=True)
)

if df.empty:

    st.error("분석할 데이터가 없습니다.")
    st.stop()


# 주차별 전체 승차인원 계산

df["전체승차인원"] = (
    df[station_columns]
    .fillna(0)
    .sum(axis=1)
)

weeks = sorted(
    df["주차"].unique().tolist()
)


# ==========================================
# 5. 데이터 기본 정보
# ==========================================

st.subheader("📊 데이터 기본 정보")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "분석 주차 수",
        f"{len(weeks)}주"
    )

with col2:
    st.metric(
        "분석 역 개수",
        f"{len(station_columns)}개"
    )

with col3:
    st.metric(
        "첫 번째 주차",
        f"{min(weeks)}주차"
    )

with col4:
    st.metric(
        "마지막 주차",
        f"{max(weeks)}주차"
    )


st.divider()


# ==========================================
# 6. 분석할 주차 범위 선택
# ==========================================

st.subheader("🔎 분석할 주차 범위 선택")

col1, col2 = st.columns(2)

with col1:

    start_week = st.selectbox(
        "시작 주차",
        weeks,
        index=0,
        format_func=lambda x: f"{x}주차"
    )

with col2:

    end_week = st.selectbox(
        "마지막 주차",
        weeks,
        index=len(weeks) - 1,
        format_func=lambda x: f"{x}주차"
    )


# 선택 범위가 잘못된 경우 안내

if start_week > end_week:

    st.warning(
        "시작 주차가 마지막 주차보다 클 수 없습니다. "
        "주차를 다시 선택해주세요."
    )

    st.stop()


# 분석 버튼

analyze_button = st.button(
    "🎈 선택한 주차 분석하기",
    type="primary",
    use_container_width=True
)

if analyze_button:

    st.balloons()

    st.success(
        f"🎉 {start_week}주차부터 {end_week}주차까지 분석을 시작합니다!"
    )


# 선택 범위 데이터

selected_df = df[
    (df["주차"] >= start_week)
    & (df["주차"] <= end_week)
].copy()

if selected_df.empty:

    st.warning("선택한 주차에 데이터가 없습니다.")
    st.stop()


st.divider()


# ==========================================
# 7. 선택 기간의 역별 승차인원 분석
# ==========================================

st.subheader("🚉 역별 승차인원 분석")

station_totals = (
    selected_df[station_columns]
    .sum()
    .sort_values(ascending=False)
)

station_result = (
    station_totals
    .reset_index()
)

station_result.columns = [
    "역명",
    "승차인원"
]

station_result["승차인원"] = (
    station_result["승차인원"]
    .fillna(0)
)


if len(station_result) > 0:

    top_station = station_result.iloc[0]

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "🏆 승차인원이 가장 많은 역",
            top_station["역명"]
        )

    with col2:

        st.metric(
            "누적 승차인원",
            f"{top_station['승차인원']:,.0f}명"
        )


# TOP 10 그래프

top10 = station_result.head(10).sort_values(
    "승차인원",
    ascending=True
)

fig_station = px.bar(
    top10,
    x="승차인원",
    y="역명",
    orientation="h",
    text="승차인원",
    title="선택 기간 역별 승차인원 TOP 10",
    color="승차인원",
    color_continuous_scale="Blues"
)

fig_station.update_traces(
    texttemplate="%{text:,.0f}명",
    textposition="outside"
)

fig_station.update_layout(
    height=500,
    xaxis_title="승차인원(명)",
    yaxis_title="역명",
    coloraxis_showscale=False,
    template="plotly_white"
)

st.plotly_chart(
    fig_station,
    use_container_width=True
)


# 역별 전체 데이터 표

with st.expander("📋 역별 승차인원 전체 보기"):

    display_station = station_result.copy()

    display_station["승차인원"] = (
        display_station["승차인원"]
        .map(lambda x: f"{x:,.0f}명")
    )

    st.dataframe(
        display_station,
        use_container_width=True,
        hide_index=True
    )


st.divider()


# ==========================================
# 8. 주차별 전체 승차인원 분석
# ==========================================

st.subheader("📅 주차별 전체 승차인원 분석")

weekly_result = selected_df[
    ["주차", "전체승차인원"]
].copy()


if not weekly_result.empty:

    max_week_row = weekly_result.loc[
        weekly_result["전체승차인원"].idxmax()
    ]

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "🏆 승차인원이 가장 많았던 주차",
            f"{int(max_week_row['주차'])}주차"
        )

    with col2:

        st.metric(
            "해당 주차 승차인원",
            f"{max_week_row['전체승차인원']:,.0f}명"
        )


fig_weekly = px.bar(
    weekly_result,
    x="주차",
    y="전체승차인원",
    text="전체승차인원",
    title="주차별 전체 승차인원",
    color="전체승차인원",
    color_continuous_scale="Blues"
)

fig_weekly.update_traces(
    texttemplate="%{text:,.0f}",
    textposition="outside"
)

fig_weekly.update_layout(
    height=500,
    xaxis_title="주차",
    yaxis_title="전체 승차인원(명)",
    xaxis=dict(
        tickmode="linear",
        dtick=1
    ),
    coloraxis_showscale=False,
    template="plotly_white"
)

st.plotly_chart(
    fig_weekly,
    use_container_width=True
)


st.divider()


# ==========================================
# 9. 역별·주차별 히트맵
# ==========================================

st.subheader("🔥 역별·주차별 승차인원 히트맵")

st.write(
    "색이 진할수록 해당 역의 승차인원이 많다는 뜻입니다."
)

heatmap_df = selected_df.set_index("주차")[
    station_columns
].T

fig_heatmap = go.Figure(
    data=go.Heatmap(
        z=heatmap_df.values,
        x=heatmap_df.columns.tolist(),
        y=heatmap_df.index.tolist(),
        colorscale="YlOrRd",
        colorbar=dict(
            title="승차인원"
        ),
        hovertemplate=(
            "역명: %{y}<br>"
            "주차: %{x}주차<br>"
            "승차인원: %{z:,.0f}명"
            "<extra></extra>"
        )
    )
)

fig_heatmap.update_layout(
    title="역별·주차별 승차인원 분포",
    xaxis_title="주차",
    yaxis_title="역명",
    height=max(500, len(station_columns) * 22),
    template="plotly_white"
)

st.plotly_chart(
    fig_heatmap,
    use_container_width=True
)


st.divider()


# ==========================================
# 10. 이상치 탐지
# ==========================================

st.subheader("🚨 이상치 탐지")

st.write(
    "IQR 방식으로 평소 주차별 승차인원과 차이가 큰 주차를 찾아봅니다."
)

# 선택한 주차 범위에서 분석

q1 = weekly_result["전체승차인원"].quantile(0.25)
q3 = weekly_result["전체승차인원"].quantile(0.75)

iqr = q3 - q1

lower_limit = q1 - 1.5 * iqr
upper_limit = q3 + 1.5 * iqr

anomaly_df = weekly_result.copy()

anomaly_df["이상치여부"] = (
    (anomaly_df["전체승차인원"] < lower_limit)
    | (anomaly_df["전체승차인원"] > upper_limit)
)

fig_anomaly = go.Figure()

fig_anomaly.add_trace(
    go.Scatter(
        x=anomaly_df["주차"],
        y=anomaly_df["전체승차인원"],
        mode="lines+markers",
        name="주차별 승차인원",
        line=dict(width=3),
        marker=dict(size=9)
    )
)

outlier_rows = anomaly_df[
    anomaly_df["이상치여부"]
]

if not outlier_rows.empty:

    fig_anomaly.add_trace(
        go.Scatter(
            x=outlier_rows["주차"],
            y=outlier_rows["전체승차인원"],
            mode="markers",
            name="이상치 후보",
            marker=dict(
                color="red",
                size=14,
                symbol="x"
            )
        )
    )

fig_anomaly.add_hline(
    y=upper_limit,
    line_dash="dash",
    annotation_text="상한 기준"
)

fig_anomaly.add_hline(
    y=lower_limit,
    line_dash="dash",
    annotation_text="하한 기준"
)

fig_anomaly.update_layout(
    title="IQR 방식 이상치 탐지 결과",
    xaxis_title="주차",
    yaxis_title="전체 승차인원(명)",
    xaxis=dict(
        tickmode="linear",
        dtick=1
    ),
    template="plotly_white",
    height=450
)

st.plotly_chart(
    fig_anomaly,
    use_container_width=True
)


if outlier_rows.empty:

    st.success(
        "선택한 기간에서 IQR 기준 이상치 후보가 발견되지 않았습니다."
    )

else:

    st.warning(
        f"이상치 후보가 {len(outlier_rows)}개 발견되었습니다."
    )

    anomaly_display = outlier_rows.copy()

    anomaly_display["전체승차인원"] = (
        anomaly_display["전체승차인원"]
        .map(lambda x: f"{x:,.0f}명")
    )

    anomaly_display["주차"] = (
        anomaly_display["주차"]
        .map(lambda x: f"{x}주차")
    )

    st.dataframe(
        anomaly_display,
        use_container_width=True,
        hide_index=True
    )


st.caption(
    "※ 이상치 후보는 통계적으로 다른 주차와 차이가 크다는 뜻이며, "
    "반드시 데이터 오류라는 의미는 아닙니다."
)


st.divider()


# ==========================================
# 11. AI 알고리즘 3가지로 미래 승차인원 예측
# ==========================================

st.subheader("🤖 AI 알고리즘 3가지로 미래 승차인원 예측")

st.write(
    "과거 주차별 전체 승차인원을 학습하여 "
    "다음 주차들의 승차인원을 예측합니다."
)


# 학습 데이터

model_df = df[
    ["주차", "전체승차인원"]
].dropna().copy()

X = model_df[["주차"]]
y = model_df["전체승차인원"]


# 예측 주차 수 선택

forecast_period = st.slider(
    "미래 몇 주까지 예측할까요?",
    min_value=1,
    max_value=12,
    value=4
)


# 사용할 알고리즘

models = {
    "선형회귀": LinearRegression(),

    "랜덤포레스트": RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        max_depth=4
    ),

    "의사결정나무": DecisionTreeRegressor(
        max_depth=4,
        random_state=42
    )
}


# ==========================================
# 12. 모델 성능 비교
# ==========================================

st.markdown("### 📈 알고리즘 성능 비교")

# 시간 순서를 유지한 학습·검증 분리
# 앞부분으로 학습하고 뒷부분으로 검증합니다.

if len(model_df) >= 8:

    split_index = max(
        2,
        int(len(model_df) * 0.8)
    )

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    evaluation_results = []

    for model_name, model in models.items():

        evaluation_model = model.__class__(
            **model.get_params()
        )

        evaluation_model.fit(
            X_train,
            y_train
        )

        test_prediction = evaluation_model.predict(
            X_test
        )

        mae = mean_absolute_error(
            y_test,
            test_prediction
        )

        if len(y_test) >= 2:

            r2 = r2_score(
                y_test,
                test_prediction
            )

        else:

            r2 = np.nan

        evaluation_results.append({
            "알고리즘": model_name,
            "MAE": mae,
            "R²": r2
        })

    evaluation_df = pd.DataFrame(
        evaluation_results
    )

    evaluation_df = evaluation_df.sort_values(
        "MAE",
        ascending=True
    ).reset_index(drop=True)

    best_model_name = evaluation_df.iloc[0]["알고리즘"]

    st.dataframe(
        evaluation_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "MAE": st.column_config.NumberColumn(
                "MAE (명)",
                format="%,.0f"
            ),
            "R²": st.column_config.NumberColumn(
                "R²",
                format="%.3f"
            )
        }
    )

    st.info(
        f"검증 데이터에서 MAE가 가장 낮은 알고리즘은 "
        f"**{best_model_name}**입니다. "
        "MAE는 예측값과 실제값의 평균적인 차이를 나타내며, "
        "낮을수록 오차가 작습니다."
    )

else:

    evaluation_df = pd.DataFrame()

    st.info(
        "모델 성능 비교를 위해서는 최소 8개 이상의 주차 데이터가 필요합니다."
    )


# ==========================================
# 13. 세 가지 알고리즘 학습 및 미래 예측
# ==========================================

future_weeks = np.arange(
    int(model_df["주차"].max()) + 1,
    int(model_df["주차"].max()) + forecast_period + 1
)

future_X = pd.DataFrame({
    "주차": future_weeks
})

prediction_results = pd.DataFrame({
    "주차": future_weeks
})

fitted_models = {}

for model_name, model in models.items():

    model.fit(
        X,
        y
    )

    predictions = model.predict(
        future_X
    )

    prediction_results[model_name] = np.maximum(
        predictions,
        0
    )

    fitted_models[model_name] = model


# ==========================================
# 14. 예측 결과 한눈에 보기
# ==========================================

st.markdown("### 🔮 알고리즘별 미래 예측 결과")

st.write(
    "같은 미래 주차에 대해 세 가지 알고리즘이 예측한 값을 비교할 수 있습니다."
)

display_prediction = prediction_results.copy()

display_prediction["주차"] = (
    display_prediction["주차"]
    .map(lambda x: f"{x}주차")
)

for column in models.keys():

    display_prediction[column] = (
        display_prediction[column]
        .map(lambda x: f"{x:,.0f}명")
    )

st.dataframe(
    display_prediction,
    use_container_width=True,
    hide_index=True
)


# ==========================================
# 15. 알고리즘별 예측 그래프
# ==========================================

fig_prediction = go.Figure()


# 실제 과거 데이터

fig_prediction.add_trace(
    go.Scatter(
        x=model_df["주차"],
        y=model_df["전체승차인원"],
        mode="lines+markers",
        name="실제 승차인원",
        line=dict(
            color="black",
            width=3
        ),
        marker=dict(size=7)
    )
)


# 알고리즘별 예측 데이터

model_colors = {
    "선형회귀": "#E74C3C",
    "랜덤포레스트": "#2980B9",
    "의사결정나무": "#27AE60"
}

for model_name in models.keys():

    fig_prediction.add_trace(
        go.Scatter(
            x=prediction_results["주차"],
            y=prediction_results[model_name],
            mode="lines+markers",
            name=f"{model_name} 예측",
            line=dict(
                color=model_colors[model_name],
                width=3,
                dash="dash"
            ),
            marker=dict(size=9)
        )
    )


fig_prediction.update_layout(
    title="실제 승차인원과 알고리즘별 미래 예측 비교",
    xaxis_title="주차",
    yaxis_title="전체 승차인원(명)",
    xaxis=dict(
        tickmode="linear",
        dtick=1
    ),
    height=550,
    hovermode="x unified",
    template="plotly_white",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

st.plotly_chart(
    fig_prediction,
    use_container_width=True
)


# ==========================================
# 16. 알고리즘 설명
# ==========================================

st.markdown("### 🧠 사용한 알고리즘 알아보기")

with st.expander("1️⃣ 선형회귀 (Linear Regression)"):

    st.write(
        """
        과거 주차와 승차인원 사이의 관계를 직선으로 표현하는 방법입니다.

        - 과거 데이터의 전체적인 증가·감소 추세를 학습합니다.
        - 일정한 추세가 이어진다고 가정하여 미래 값을 예측합니다.
        - 장기적인 추세를 확인하기에 좋습니다.
        """
    )

with st.expander("2️⃣ 랜덤포레스트 (Random Forest)"):

    st.write(
        """
        여러 개의 의사결정나무를 만들고,
        각 나무의 예측 결과를 종합하는 방법입니다.

        - 여러 나무의 결과를 평균하여 예측합니다.
        - 데이터의 복잡한 패턴을 학습할 수 있습니다.
        - 학습한 범위를 벗어난 미래 추세를 잘 따라가지 못할 수 있습니다.
        """
    )

with st.expander("3️⃣ 의사결정나무 (Decision Tree)"):

    st.write(
        """
        데이터를 여러 조건으로 나누어 예측하는 방법입니다.

        - 데이터의 특징을 조건별로 구분합니다.
        - 나무 구조로 예측 과정을 이해하기 쉽습니다.
        - 학습 데이터에 따라 예측값이 크게 달라질 수 있습니다.
        """
    )


st.warning(
    "📌 주의: 미래 예측은 과거 주차별 승차인원만을 이용한 참고용 추정치입니다. "
    "실제 승차인원은 공휴일, 계절, 운행 편수, 행사 등의 영향을 받을 수 있습니다. "
    "특히 랜덤포레스트와 의사결정나무는 학습 데이터 범위 밖의 미래 추세를 "
    "정확히 연장하지 못할 수 있습니다."
)


st.divider()


# ==========================================
# 17. 분석 결과 다운로드
# ==========================================

st.subheader("📥 분석 결과 다운로드")

# 역별 분석 결과
station_csv = station_result.to_csv(
    index=False
).encode("utf-8-sig")


# 주차별 분석 결과
weekly_csv = weekly_result.to_csv(
    index=False
).encode("utf-8-sig")


# 예측 결과
prediction_csv = prediction_results.to_csv(
    index=False
).encode("utf-8-sig")


col1, col2, col3 = st.columns(3)

with col1:

    st.download_button(
        label="⬇️ 역별 승차인원 다운로드",
        data=station_csv,
        file_name="SRT_역별_승차인원.csv",
        mime="text/csv",
        use_container_width=True
    )

with col2:

    st.download_button(
        label="⬇️ 주차별 승차인원 다운로드",
        data=weekly_csv,
        file_name="SRT_주차별_승차인원.csv",
        mime="text/csv",
        use_container_width=True
    )

with col3:

    st.download_button(
        label="⬇️ 미래 예측 결과 다운로드",
        data=prediction_csv,
        file_name="SRT_미래_승차인원_예측.csv",
        mime="text/csv",
        use_container_width=True
    )


st.divider()


# ==========================================
# 18. 마무리
# ==========================================

st.success(
    "🎉 SRT 주차별 승차인원 분석이 완료되었습니다!"
)

st.caption(
    "📊 데이터 출처: (주)에스알_주별 평균 승차인원_20250930.csv"
)

st.caption(
    "분석 기능: pandas · Plotly · scikit-learn"
)
