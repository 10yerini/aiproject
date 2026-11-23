
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


# ==========================================
# 1. 페이지 설정
# ==========================================

st.set_page_config(
    page_title="아이리스 품종 예측",
    page_icon="🌸",
    layout="wide"
)

st.title("🌸 아이리스 품종 예측 프로그램")
st.write("꽃잎의 길이와 너비를 입력하고 품종을 예측해 보세요!")

st.divider()


# ==========================================
# 2. 한글 폰트 설정
# ==========================================

# koreanize_matplotlib 설치 시 한글 폰트 자동 설정
try:
    import koreanize_matplotlib
    koreanize_matplotlib.koreanize()
except ImportError:
    # 설치되지 않은 경우 시스템에 있는 한글 폰트 탐색
    available_fonts = {f.name for f in fm.fontManager.ttflist}

    font_candidates = [
        "Malgun Gothic",
        "NanumGothic",
        "AppleGothic"
    ]

    for font in font_candidates:
        if font in available_fonts:
            plt.rcParams["font.family"] = font
            break

plt.rcParams["axes.unicode_minus"] = False


# ==========================================
# 3. CSV 파일 불러오기
# ==========================================

@st.cache_data
def load_data(file_path):

    encodings = [
        "utf-8-sig",
        "cp949",
        "euc-kr",
        "utf-8"
    ]

    for encoding in encodings:
        try:
            data = pd.read_csv(
                file_path,
                encoding=encoding
            )
            return data

        except UnicodeDecodeError:
            continue

    raise ValueError("CSV 파일의 인코딩을 확인해 주세요.")


# 현재 파일 위치 확인
current_folder = Path(__file__).resolve().parent

# CSV 파일 경로
file_path = current_folder / "Iris.csv"

# pages 폴더 상위 폴더에서도 찾기
if not file_path.exists():
    file_path = current_folder.parent / "Iris.csv"


if not file_path.exists():
    st.error(
        "Iris.csv 파일을 찾을 수 없습니다. "
        "pages 폴더의 상위 폴더에 CSV 파일을 넣어 주세요."
    )
    st.stop()


try:
    df = load_data(str(file_path))

except Exception as e:
    st.error(f"CSV 파일을 읽는 중 오류가 발생했습니다: {e}")
    st.stop()


# ==========================================
# 4. 컬럼명 확인 및 데이터 정리
# ==========================================

# 앞뒤 공백 제거
df.columns = df.columns.astype(str).str.strip()

# 한글 컬럼명과 영어 컬럼명 모두 지원
column_aliases = {
    "꽃잎 길이": [
        "꽃잎 길이", "PetalLengthCm", "Petal Length",
        "petal_length", "petal length"
    ],
    "꽃잎 너비": [
        "꽃잎 너비", "PetalWidthCm", "Petal Width",
        "petal_width", "petal width"
    ],
    "품종": [
        "품종", "Species", "species", "variety"
    ]
}

rename_dict = {}

for standard_name, aliases in column_aliases.items():
    for alias in aliases:
        if alias in df.columns:
            rename_dict[alias] = standard_name
            break

df = df.rename(columns=rename_dict)

required_columns = [
    "꽃잎 길이",
    "꽃잎 너비",
    "품종"
]

if not all(col in df.columns for col in required_columns):
    st.error("CSV 파일의 컬럼명을 확인해 주세요.")
    st.write("현재 CSV 컬럼:", df.columns.tolist())
    st.stop()


# 숫자형 데이터 변환
df["꽃잎 길이"] = pd.to_numeric(
    df["꽃잎 길이"], errors="coerce"
)

df["꽃잎 너비"] = pd.to_numeric(
    df["꽃잎 너비"], errors="coerce"
)

df = df.dropna(subset=required_columns).copy()

if len(df) < 5:
    st.error("학습에 필요한 데이터가 부족합니다.")
    st.stop()


# ==========================================
# 5. 데이터 범위 확인
# ==========================================

X = df[["꽃잎 길이", "꽃잎 너비"]]
y = df["품종"].astype(str)

min_length = float(X["꽃잎 길이"].min())
max_length = float(X["꽃잎 길이"].max())

min_width = float(X["꽃잎 너비"].min())
max_width = float(X["꽃잎 너비"].max())


# ==========================================
# 6. 품종 이름 한글 변환
# ==========================================

species_korean = {
    "Iris-setosa": "아이리스 세토사",
    "Iris-versicolor": "아이리스 버시컬러",
    "Iris-virginica": "아이리스 버지니카",
    "setosa": "아이리스 세토사",
    "versicolor": "아이리스 버시컬러",
    "virginica": "아이리스 버지니카"
}

def korean_name(name):
    return species_korean.get(name, name)


# ==========================================
# 7. KNN 머신러닝 모델 학습
# ==========================================

model = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=5)
)

model.fit(X, y)


# ==========================================
# 8. 데이터 기본 정보
# ==========================================

st.subheader("📊 아이리스 데이터 정보")

col1, col2, col3 = st.columns(3)

col1.metric("전체 데이터", f"{len(df)}개")
col2.metric("꽃잎 길이 범위", f"{min_length}~{max_length}cm")
col3.metric("꽃잎 너비 범위", f"{min_width}~{max_width}cm")


# ==========================================
# 9. 사용자 입력
# ==========================================

st.divider()

st.subheader("✏️ 꽃잎 크기 입력하기")

st.info(
    f"""
    입력할 수 있는 값의 범위

    🌷 꽃잎 길이: {min_length} ~ {max_length} cm

    🌷 꽃잎 너비: {min_width} ~ {max_width} cm
    """
)

col1, col2 = st.columns(2)

with col1:
    petal_length = st.number_input(
        "꽃잎 길이 (cm)",
        min_value=min_length,
        max_value=max_length,
        value=min(5.0, max_length),
        step=0.1,
        format="%.1f"
    )

with col2:
    petal_width = st.number_input(
        "꽃잎 너비 (cm)",
        min_value=min_width,
        max_value=max_width,
        value=min(1.5, max_width),
        step=0.1,
        format="%.1f"
    )


# ==========================================
# 10. 품종 예측 버튼
# ==========================================

if st.button(
    "🔍 아이리스 품종 예측하기",
    use_container_width=True
):

    input_data = pd.DataFrame(
        [[petal_length, petal_width]],
        columns=["꽃잎 길이", "꽃잎 너비"]
    )

    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]

    st.divider()

    st.subheader("🎉 예측 결과")

    st.success(
        f"예측된 품종은 [{korean_name(prediction)}] 입니다!"
    )

    # --------------------------------------
    # 예측 확률
    # --------------------------------------

    st.write("📈 품종별 예측 결과")

    probability_df = pd.DataFrame({
        "품종": [
            korean_name(name)
            for name in model.classes_
        ],
        "예측 확률 (%)": probabilities * 100
    })

    st.bar_chart(
        probability_df.set_index("품종"),
        y="예측 확률 (%)"
    )

    for _, row in probability_df.iterrows():
        st.write(
            f"{row['품종']}: {row['예측 확률 (%)']:.1f}%"
        )

    # ======================================
    # 11. 산점도 그래프
    # ======================================

    st.divider()

    st.subheader("📍 아이리스 꽃잎 분포 그래프")

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {
        "Iris-setosa": "blue",
        "Iris-versicolor": "green",
        "Iris-virginica": "red",
        "setosa": "blue",
        "versicolor": "green",
        "virginica": "red"
    }

    # 품종별 데이터 표시
    for species, group in df.groupby("품종"):

        ax.scatter(
            group["꽃잎 길이"],
            group["꽃잎 너비"],
            color=colors.get(species, "gray"),
            label=korean_name(species),
            alpha=0.65,
            s=65
        )

    # 사용자 입력 위치 표시
    ax.scatter(
        petal_length,
        petal_width,
        color="yellow",
        edgecolors="black",
        marker="*",
        s=500,
        linewidths=1.5,
        label="내가 입력한 값",
        zorder=10
    )

    # 별표 위치 설명
    ax.annotate(
        "내가 입력한 위치",
        (petal_length, petal_width),
        xytext=(12, 12),
        textcoords="offset points",
        fontsize=11,
        fontweight="bold"
    )

    ax.set_xlabel("꽃잎 길이 (cm)", fontsize=13)
    ax.set_ylabel("꽃잎 너비 (cm)", fontsize=13)

    ax.set_title(
        "아이리스 품종별 꽃잎 크기 분포",
        fontsize=16
    )

    ax.legend()
    ax.grid(alpha=0.3)

    st.pyplot(fig)

    plt.close(fig)

    # ======================================
    # 12. 알고리즘 설명
    # ======================================

    st.divider()

    st.subheader("🧠 KNN 알고리즘 알아보기")

    st.write(
        """
        KNN은 K-최근접 이웃이라는 분류 알고리즘입니다.

        **① 입력값 확인**
        
        사용자가 입력한 꽃잎 길이와 너비를 확인합니다.

        **② 가까운 데이터 찾기**
        
        기존 아이리스 데이터에서 입력값과 가장 가까운
        데이터 5개를 찾습니다.

        **③ 품종 비교**
        
        가까운 5개의 데이터가 어떤 품종인지 비교합니다.

        **④ 최종 예측**
        
        가장 많이 등장한 품종을 최종 결과로 선택합니다.
        """
    )

    st.caption(
        "※ 예측 확률은 가장 가까운 5개 데이터에서 "
        "각 품종이 차지하는 비율입니다."
    )

else:

    st.info(
        "꽃잎 길이와 너비를 입력한 다음 "
        "예측하기 버튼을 눌러 주세요."
    )


# ==========================================
# 13. 프로그램 하단 안내
# ==========================================

st.divider()

st.caption(
    "🌸 아이리스 품종 예측 프로그램 | "
    "Pandas + Scikit-learn + Streamlit"
)
