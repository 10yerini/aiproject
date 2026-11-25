
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

st.title("🚄 SRT 주차별 승차인원 분석")
st.markdown(
    "원하는 주차를 선택하고 가장 많이 이용한 역과 "
    "승차인원 추세를 분석해 보세요!"
)

st.divider()


# ==========================================
# 2. CSV 파일 불러오기
# ==========================================

@st.cache_data
def load_data(file_bytes=None):

    if file_bytes is not None:
        import io
        return pd.read_csv(
            io.BytesIO(file_bytes),
            encoding="cp949"
        )

    # 현재 py 파일은 pages 폴더에 있음
    base_dir = Path(__file__).resolve().parent.parent

    file_path = (
        base_dir /
        "(주)에스알_주별 평균 승차인원_20250930.csv"
    )

    if not file_path.exists():
        return None

    return pd.read_csv(
        file_path,
        encoding="cp949"
    )


uploaded_file = st.file_uploader(
    "CSV 파일 업로드 (필요한 경우)",
    type=["csv"]
)

try:
    if uploaded_file is not None:
        df = load_data(uploaded_file.getvalue())
    else:
        df = load_data()

except Exception as e:
    st.error(f"CSV 파일을 읽는 중 오류가 발생했습니다: {e}")
    st.stop()


if df is None:
    st.error(
        "CSV 파일을 찾을 수 없습니다. "
        "pages 폴더의 상위 폴더에 CSV 파일을 넣어 주세요."
    )
    st.stop()


# ==========================================
# 3. 데이터 전처리
# ==========================================

df.columns = df.columns.str.strip()

if "주차구분" not in df.columns:
    st.error("주차구분 컬럼을 찾을 수 없습니다.")
    st.stop()

# 주차 숫자 추출
df["주차"] = (
    df["주차구분"]
    .astype(str)
    .str.extract(r"(\d+)\s*주차")[0]
)

df["주차"] = pd.to_numeric(
    df["주차"],
    errors="coerce"
)

# 역 이름 컬럼
station_columns = [
    col for col in df.columns
    if col not in ["주차구분", "주차"]
]

# 숫자로 변환
for col in station_columns:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    ).fillna(0)

df = df.dropna(subset=["주차"])
df["주차"] = df["주차"].astype(int)

df = df.sort_values("주차")

if df.empty:
    st.error("분석할 데이터가 없습니다.")
    st.stop()

# 주차별 전체 승차인원
df["전체승차인원"] = df[station_columns].sum(axis=1)

st.success(
    f"데이터 불러오기 성공! "
    f"전체 {len(df)}개 주차, "
    f"{len(station_columns)}개 역"
)


# ==========================================
# 4. 주차 범위 선택
# ==========================================

st.subheader("📅 분석할 주차 선택")

available_weeks = sorted(df["주차"].unique())

min_week = int(min(available_weeks))
max_week = int(max(available_weeks))

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


# 선택한 기간의 데이터
selected_df = df[
    (df["주차"] >= start_week) &
    (df["주차"] <= end_week)
].copy()

if selected_df.empty:
    st.warning("선택한 주차에 데이터가 없습니다.")
    st.stop()

st.info(
    f"현재 분석 범위: {start_week}주차 ~ {end_week}주차 "
    f"({len(selected_df)}개 주차)"
)


# ==========================================
# 5. 역별 승차인원 분석
# ==========================================

st.divider()
st.subheader("🏆 가장 많이 이용한 역")

# 역별 선택 기간 승차인원 합계
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

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "🥇 승차인원 1위 역",
        top_station
    )

with col2:
    st.metric(
        "🚄 1위 역 승차인원",
        f"{top_passengers:,}명"
    )

with col3:
    st.metric(
        "📊 전체 승차인원 합계",
        f"{total_passengers:,}명"
    )


# 역별 TOP 10
st.markdown("### 역별 승차인원 TOP 10")

top10 = station_total.head(10).sort_values()

fig_station = go.Figure()

fig_station.add_trace(
    go.Bar(
        x=top10.values,
        y=top10.index,
        orientation="h",
        text=[
            f"{int(v):,}명"
            for v in top10.values
        ],
        textposition="outside",
        marker_color="#3498DB"
    )
)

fig_station.update_layout(
    title="선택한 주차 범위의 역별 승차인원",
    xaxis_title="승차인원",
    yaxis_title="역 이름",
    height=500,
    template="plotly_white"
)

st.plotly_chart(
    fig_station,
    use_container_width=True
)


# ==========================================
# 6. 가장 많이 이용한 주차
# ==========================================

st.divider()
st.subheader("📈 가장 많이 이용한 주차")

max_index = selected_df[
    "전체승차인원"
].idxmax()

max_row = selected_df.loc[max_index]

max_week_number = int(max_row["주차"])
max_week_passengers = int(
    max_row["전체승차인원"]
)

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "🏆 승차인원이 가장 많은 주차",
        f"{max_week_number}주차"
    )

with col2:
    st.metric(
        "👥 해당 주차 승차인원",
        f"{max_week_passengers:,}명"
    )


# 주차별 전체 승차인원 그래프
fig_week = go.Figure()

fig_week.add_trace(
    go.Bar(
        x=selected_df["주차"],
        y=selected_df["전체승차인원"],
        name="주차별 승차인원",
        marker_color="#5DADE2"
    )
)

fig_week.add_trace(
    go.Scatter(
        x=[max_week_number],
        y=[max_week_passengers],
        mode="markers+text",
        name="최다 승차 주차",
        marker=dict(
            color="red",
            size=15,
            symbol="star"
        ),
        text=["최다 승차"],
        textposition="top center"
    )
)

fig_week.update_layout(
    title="주차별 전체 승차인원",
    xaxis_title="주차",
    yaxis_title="승차인원 (명)",
    xaxis=dict(dtick=1),
    template="plotly_white",
    height=450
)

st.plotly_chart(
    fig_week,
    use_container_width=True
)


# ==========================================
# 7. 알고리즘 분석
# ==========================================

st.divider()
st.subheader("🤖 인공지능 알고리즘 분석")

st.write(
    "선택한 주차 범위의 승차인원 데이터를 "
    "세 가지 머신러닝 알고리즘으로 분석합니다."
)

if len(selected_df) < 4:
    st.warning(
        "알고리즘 비교를 위해 최소 4개 주차 이상의 "
        "데이터를 선택해 주세요."
    )
    st.stop()


# X: 주차
X = selected_df[["주차"]].values

# y: 전체 승차인원
y = selected_df["전체승차인원"].values


# ==========================================
# 8. 세 가지 알고리즘 생성
# ==========================================

models = {
    "선형회귀": LinearRegression(),

    "랜덤 포레스트": RandomForestRegressor(
        n_estimators=100,
        max_depth=3,
        random_state=42
    ),

    "의사결정나무": DecisionTreeRegressor(
        max_depth=3,
        random_state=42
    )
}


# 그래프용 주차
graph_weeks = np.arange(
    start_week,
    end_week + 1
).reshape(-1, 1)


results = []
predictions = {}

for name, model in models.items():

    model.fit(X, y)

    # 학습 데이터 예측
    train_pred = model.predict(X)

    # 그래프용 예측
    graph_pred = model.predict(graph_weeks)

    # 성능 측정
    mae = mean_absolute_error(
        y,
        train_pred
    )

    r2 = r2_score(
        y,
        train_pred
    )

    results.append({
        "알고리즘": name,
        "평균절대오차(MAE)": round(mae, 2),
        "결정계수(R²)": round(r2, 3)
    })

    predictions[name] = graph_pred


# ==========================================
# 9. 알고리즘별 그래프
# ==========================================

fig_model = go.Figure()

# 실제 데이터
fig_model.add_trace(
    go.Scatter(
        x=selected_df["주차"],
        y=y,
        mode="lines+markers",
        name="실제 승차인원",
        line=dict(
            color="black",
            width=3
        )
    )
)

colors = [
    "#E74C3C",
    "#3498DB",
    "#2ECC71"
]

for (name, pred), color in zip(
    predictions.items(),
    colors
):

    fig_model.add_trace(
        go.Scatter(
            x=graph_weeks.flatten(),
            y=pred,
            mode="lines",
            name=name,
            line=dict(
                color=color,
                width=2,
                dash="dash"
            )
        )
    )


fig_model.update_layout(
    title="실제 승차인원과 알고리즘별 분석 비교",
    xaxis_title="주차",
    yaxis_title="승차인원 (명)",
    xaxis=dict(dtick=1),
    template="plotly_white",
    height=500,
    hovermode="x unified"
)

st.plotly_chart(
    fig_model,
    use_container_width=True
)


# ==========================================
# 10. 알고리즘 성능 비교표
# ==========================================

st.markdown("### 📊 알고리즘 성능 비교")

result_df = pd.DataFrame(results)

st.dataframe(
    result_df,
    use_container_width=True,
    hide_index=True
)

st.caption(
    "MAE는 예측값과 실제값의 평균적인 차이이며 "
    "낮을수록 좋습니다. R²는 학습 데이터에 대한 "
    "설명력을 나타내며 높을수록 좋습니다. "
    "두 지표 모두 선택한 데이터에 대한 학습 결과이므로 "
    "새로운 주차에 대한 예측 성능과는 다를 수 있습니다."
)


# ==========================================
# 11. 알고리즘 설명
# ==========================================

st.divider()
st.subheader("📚 알고리즘 알아보기")

with st.expander("① 선형회귀 (Linear Regression)"):

    st.write(
        "주차가 증가함에 따라 승차인원이 "
        "어떻게 변화하는지 직선 형태로 분석합니다."
    )

    st.latex("y = ax + b")

    st.write(
        "전체적인 증가 또는 감소 추세를 "
        "파악하는 데 유용합니다."
    )


with st.expander("② 랜덤 포레스트 (Random Forest)"):

    st.write(
        "여러 개의 의사결정나무를 만들어 "
        "각 결과를 종합하는 알고리즘입니다."
    )

    st.write(
        "복잡한 데이터 패턴을 학습할 수 있지만 "
        "데이터가 적으면 결과가 불안정할 수 있습니다."
    )


with st.expander("③ 의사결정나무 (Decision Tree)"):

    st.write(
        "데이터를 여러 구간으로 나누어 "
        "승차인원 패턴을 분석합니다."
    )

    st.write(
        "특정 주차 구간에서 승차인원이 "
        "어떻게 달라지는지 살펴보기 좋습니다."
    )


# ==========================================
# 12. 분석 결과 요약
# ==========================================

st.divider()

st.subheader("📝 분석 결과 요약")

st.markdown(
    f"""
    - 분석 기간: **{start_week}주차 ~ {end_week}주차**
    - 가장 많이 이용한 역: **{top_station}역**
    - 해당 역의 승차인원 합계: **{top_passengers:,}명**
    - 전체 승차인원이 가장 많은 주차: **{max_week_number}주차**
    - 해당 주차의 전체 승차인원: **{max_week_passengers:,}명**
    - 분석 알고리즘: **선형회귀, 랜덤 포레스트, 의사결정나무**
    """
)

st.success("SRT 승차인원 분석이 완료되었습니다! 🚄")

