
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# --------------------------------
# 1. 페이지 기본 설정
# --------------------------------

st.set_page_config(
    page_title="아이리스 품종 예측",
    page_icon="🌸",
    layout="wide"
)

st.title("🌸 아이리스 품종 예측 프로그램")
st.write(
    "꽃잎의 길이와 너비를 입력하면 "
    "머신러닝 알고리즘이 아이리스 품종을 예측합니다!"
)

st.divider()

# --------------------------------
# 2. CSV 파일 불러오기
# --------------------------------

file_path = Path(__file__).resolve().parent / "Iris.csv"

# pages 폴더의 상위 폴더에서 파일 찾기
if not file_path.exists():
    file_path = Path(__file__).resolve().parent.parent / "Iris.csv"

if not file_path.exists():
    st.error(
        "Iris.csv 파일을 찾을 수 없습니다. "
        "pages 폴더의 상위 폴더에 CSV 파일을 넣어 주세요."
    )
    st.stop()

# 한글 인코딩 처리
try:
    df = pd.read_csv(file_path, encoding="cp949")
except UnicodeDecodeError:
    df = pd.read_csv(file_path, encoding="utf-8-sig")

# --------------------------------
# 3. 데이터 확인
# --------------------------------

required_columns = [
    "꽃잎 길이",
    "꽃잎 너비",
    "품종"
]

if not all(col in df.columns for col in required_columns):
    st.error(
        "CSV 파일에 필요한 컬럼이 없습니다. "
        "꽃잎 길이, 꽃잎 너비, 품종 컬럼을 확인해 주세요."
    )
    st.stop()

df = df.dropna(subset=required_columns).copy()

X = df[["꽃잎 길이", "꽃잎 너비"]]
y = df["품종"]

# 입력 범위 계산
min_length = float(X["꽃잎 길이"].min())
max_length = float(X["꽃잎 길이"].max())

min_width = float(X["꽃잎 너비"].min())
max_width = float(X["꽃잎 너비"].max())

# --------------------------------
# 4. KNN 알고리즘 학습
# --------------------------------

# 표준화 + KNN 분류기
model = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=5)
)

model.fit(X, y)

# --------------------------------
# 5. 사용자 입력
# --------------------------------

st.subheader("✏️ 꽃잎 크기 입력하기")

st.info(
    f"""
    📌 입력 가능한 범위

    - 꽃잎 길이: {min_length} ~ {max_length} cm
    - 꽃잎 너비: {min_width} ~ {max_width} cm
    """
)

col1, col2 = st.columns(2)

with col1:
    petal_length = st.number_input(
        "🌷 꽃잎 길이 (cm)",
        min_value=min_length,
        max_value=max_length,
        value=5.0,
        step=0.1,
        format="%.1f"
    )

with col2:
    petal_width = st.number_input(
        "🌷 꽃잎 너비 (cm)",
        min_value=min_width,
        max_value=max_width,
        value=1.5,
        step=0.1,
        format="%.1f"
    )

# --------------------------------
# 6. 품종 예측
# --------------------------------

if st.button("🔍 아이리스 품종 예측하기", use_container_width=True):

    input_data = pd.DataFrame(
        [[petal_length, petal_width]],
        columns=["꽃잎 길이", "꽃잎 너비"]
    )

    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]

    st.divider()

    st.subheader("🎉 예측 결과")

    st.success(f"예측된 아이리스 품종: {prediction}")

    # 품종별 예측 확률
    st.write("📊 품종별 예측 확률")

    probability_df = pd.DataFrame({
        "품종": model.classes_,
        "예측 확률": probabilities
    })

    probability_df["예측 확률"] *= 100

    st.bar_chart(
        probability_df.set_index("품종")
    )

    for _, row in probability_df.iterrows():
        st.write(
            f"{row['품종']}: {row['예측 확률']:.1f}%"
        )

    # --------------------------------
    # 7. 산점도 시각화
    # --------------------------------

    st.divider()
    st.subheader("📍 꽃잎 길이와 너비의 분포")

    fig, ax = plt.subplots(figsize=(10, 6))

    # 한글 폰트 설정
    plt.rcParams["font.family"] = [
        "Malgun Gothic",
        "NanumGothic",
        "DejaVu Sans"
    ]
    plt.rcParams["axes.unicode_minus"] = False

    # 품종별 색상
    colors = {
        "Iris-setosa": "blue",
        "Iris-versicolor": "green",
        "Iris-virginica": "red"
    }

    # 기존 데이터 표시
    for species, group in df.groupby("품종"):

        ax.scatter(
            group["꽃잎 길이"],
            group["꽃잎 너비"],
            label=species,
            color=colors.get(species, "gray"),
            alpha=0.65,
            s=60
        )

    # 사용자가 입력한 위치를 노란색 별로 표시
    ax.scatter(
        petal_length,
        petal_width,
        color="yellow",
        edgecolors="black",
        marker="*",
        s=450,
        label="내가 입력한 값",
        zorder=10
    )

    # 입력 위치에 텍스트 표시
    ax.annotate(
        "내가 입력한 위치",
        (petal_length, petal_width),
        xytext=(10, 12),
        textcoords="offset points",
        fontsize=10,
        fontweight="bold"
    )

    ax.set_xlabel("꽃잎 길이 (cm)", fontsize=12)
    ax.set_ylabel("꽃잎 너비 (cm)", fontsize=12)

    ax.set_title(
        "아이리스 품종별 꽃잎 분포",
        fontsize=16
    )

    ax.legend()
    ax.grid(alpha=0.3)

    st.pyplot(fig)

    # --------------------------------
    # 8. 알고리즘 설명
    # --------------------------------

    st.divider()

    st.subheader("🧠 KNN 알고리즘이란?")

    st.write(
        """
        KNN은 K-최근접 이웃이라는 분류 알고리즘입니다.

        1. 사용자가 입력한 꽃잎 길이와 너비를 확인합니다.

        2. 기존 아이리스 데이터에서 입력값과 가까운
           데이터 5개를 찾습니다.

        3. 가장 가까운 5개의 데이터 중에서
           가장 많이 등장하는 품종을 선택합니다.

        4. 선택된 품종을 최종 예측 결과로 보여줍니다.
        """
    )

    st.caption(
        "※ 예측 확률은 KNN의 최근접 이웃 5개 중 "
        "각 품종이 차지하는 비율입니다."
    )

else:
    st.info(
        "꽃잎 길이와 너비를 입력한 후 "
        "'아이리스 품종 예측하기' 버튼을 눌러 주세요."
    )
