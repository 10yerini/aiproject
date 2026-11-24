
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


# ==========================================
# 1. 페이지 설정
# ==========================================

st.set_page_config(
    page_title="아이리스 인공지능 품종 예측",
    page_icon="🌸",
    layout="wide"
)

st.title("🌸 아이리스 인공지능 품종 예측")

st.write(
    "꽃잎의 길이와 너비를 입력하고 "
    "세 가지 머신러닝 알고리즘으로 품종을 예측해 보세요."
)

st.divider()


# ==========================================
# 2. CSV 파일 불러오기
# ==========================================

현재폴더 = Path(__file__).resolve().parent

파일경로 = 현재폴더 / "Iris.csv"

if not 파일경로.exists():
    파일경로 = 현재폴더.parent / "Iris.csv"

if not 파일경로.exists():
    st.error(
        "Iris.csv 파일을 찾을 수 없습니다. "
        "pages 폴더의 상위 폴더에 CSV 파일을 넣어 주세요."
    )
    st.stop()


@st.cache_data
def 데이터불러오기(경로):

    인코딩목록 = [
        "cp949",
        "utf-8-sig",
        "utf-8",
        "euc-kr"
    ]

    for 인코딩 in 인코딩목록:
        try:
            return pd.read_csv(
                경로,
                encoding=인코딩
            )
        except UnicodeDecodeError:
            continue

    raise ValueError("CSV 파일의 인코딩을 확인해 주세요.")


try:
    df = 데이터불러오기(str(파일경로))

except Exception as e:
    st.error(f"데이터 파일 오류: {e}")
    st.stop()


# ==========================================
# 3. 데이터 전처리
# ==========================================

df.columns = df.columns.astype(str).str.strip()

컬럼별칭 = {
    "petal_length": [
        "PetalLengthCm",
        "Petal Length",
        "petal_length",
        "꽃잎 길이",
        "꽃잎길이"
    ],

    "petal_width": [
        "PetalWidthCm",
        "Petal Width",
        "petal_width",
        "꽃잎 너비",
        "꽃잎너비"
    ],

    "species": [
        "Species",
        "species",
        "품종"
    ]
}

컬럼이름변경 = {}

for 표준이름, 별칭목록 in 컬럼별칭.items():

    for 별칭 in 별칭목록:

        if 별칭 in df.columns:

            컬럼이름변경[별칭] = 표준이름
            break

df = df.rename(columns=컬럼이름변경)

필수컬럼 = [
    "petal_length",
    "petal_width",
    "species"
]

if not all(컬럼 in df.columns for 컬럼 in 필수컬럼):

    st.error("CSV 파일의 컬럼명을 확인해 주세요.")

    st.write(
        "현재 CSV 컬럼:",
        df.columns.tolist()
    )

    st.stop()


df["petal_length"] = pd.to_numeric(
    df["petal_length"],
    errors="coerce"
)

df["petal_width"] = pd.to_numeric(
    df["petal_width"],
    errors="coerce"
)

df["species"] = df["species"].astype(str).str.strip()

df = df.dropna(
    subset=필수컬럼
).copy()

if len(df) < 5:
    st.error("학습에 필요한 데이터가 부족합니다.")
    st.stop()

if df["species"].nunique() < 2:
    st.error("품종이 두 종류 이상 있어야 분류할 수 있습니다.")
    st.stop()

X = df[
    ["petal_length", "petal_width"]
]

y = df["species"]


# ==========================================
# 4. 품종 이름 한글 변환
# ==========================================

품종이름 = {
    "Iris-setosa": "세토사",
    "Iris-versicolor": "버시컬러",
    "Iris-virginica": "버지니카",
    "setosa": "세토사",
    "versicolor": "버시컬러",
    "virginica": "버지니카"
}


def 품종한글(이름):
    return 품종이름.get(이름, 이름)


품종색상 = {
    "Iris-setosa": "blue",
    "Iris-versicolor": "green",
    "Iris-virginica": "red",
    "setosa": "blue",
    "versicolor": "green",
    "virginica": "red"
}


# ==========================================
# 5. 입력 가능한 값의 범위
# ==========================================

최소길이 = float(X["petal_length"].min())
최대길이 = float(X["petal_length"].max())

최소너비 = float(X["petal_width"].min())
최대너비 = float(X["petal_width"].max())


# ==========================================
# 6. 세 가지 알고리즘 만들기
# ==========================================

K = 5

모델들 = {

    "KNN": make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(
            n_neighbors=K
        )
    ),

    "의사결정나무": DecisionTreeClassifier(
        max_depth=4,
        random_state=42
    ),

    "랜덤 포레스트": RandomForestClassifier(
        n_estimators=100,
        max_depth=4,
        random_state=42
    )

}


# ==========================================
# 7. 알고리즘 학습
# ==========================================

for 모델 in 모델들.values():
    모델.fit(X, y)


# ==========================================
# 8. 데이터 정보
# ==========================================

st.subheader("📊 아이리스 데이터 정보")

열1, 열2, 열3 = st.columns(3)

열1.metric(
    "전체 데이터",
    f"{len(df)}개"
)

열2.metric(
    "꽃 품종",
    f"{y.nunique()}종"
)

열3.metric(
    "분류 알고리즘",
    "3개"
)

st.divider()


# ==========================================
# 9. 사용자 입력
# ==========================================

st.subheader("✏️ 꽃잎 크기 입력하기")

st.info(
    f"""
    입력 가능한 값의 범위

    🌷 꽃잎 길이: {최소길이} ~ {최대길이} cm

    🌷 꽃잎 너비: {최소너비} ~ {최대너비} cm
    """
)

열1, 열2 = st.columns(2)

with 열1:

    꽃잎길이 = st.number_input(
        "꽃잎 길이 (cm)",
        min_value=최소길이,
        max_value=최대길이,
        value=min(5.0, 최대길이),
        step=0.1,
        format="%.1f"
    )

with 열2:

    꽃잎너비 = st.number_input(
        "꽃잎 너비 (cm)",
        min_value=최소너비,
        max_value=최대너비,
        value=min(1.5, 최대너비),
        step=0.1,
        format="%.1f"
    )


# ==========================================
# 10. 알고리즘 선택
# ==========================================

st.divider()

st.subheader("🤖 예측 알고리즘 선택")

선택알고리즘 = st.selectbox(
    "사용할 분류 알고리즘을 선택하세요.",
    [
        "KNN",
        "의사결정나무",
        "랜덤 포레스트"
    ]
)

알고리즘설명 = {

    "KNN":
        "가장 가까운 데이터 5개를 찾아 "
        "다수결로 품종을 결정합니다.",

    "의사결정나무":
        "꽃잎의 길이와 너비에 관한 여러 조건을 "
        "차례대로 확인합니다.",

    "랜덤 포레스트":
        "100개의 의사결정나무가 각각 예측한 "
        "결과를 종합합니다."

}

st.info(
    f"선택한 알고리즘: {선택알고리즘}\n\n"
    f"{알고리즘설명[선택알고리즘]}"
)


# ==========================================
# 11. 예측 및 비교
# ==========================================

if st.button(
    "🔍 품종 예측 및 알고리즘 비교하기",
    use_container_width=True
):

    입력데이터 = pd.DataFrame(
        [[꽃잎길이, 꽃잎너비]],
        columns=[
            "petal_length",
            "petal_width"
        ]
    )

    예측결과 = {}
    확률결과 = {}
    비교자료 = []


    # --------------------------------------
    # 세 가지 알고리즘으로 예측
    # --------------------------------------

    for 이름, 모델 in 모델들.items():

        예측 = 모델.predict(입력데이터)[0]

        확률 = 모델.predict_proba(
            입력데이터
        )[0]

        예측결과[이름] = 예측
        확률결과[이름] = 확률

        비교자료.append({

            "알고리즘": 이름,

            "예측 품종": 품종한글(예측),

            "최고 예측 확률(%)": round(
                max(확률) * 100,
                1
            )

        })


    # ======================================
    # 12. 선택한 알고리즘의 결과
    # ======================================

    st.divider()

    st.subheader("🎯 선택한 알고리즘의 예측 결과")

    선택예측 = 예측결과[선택알고리즘]

    st.success(
        f"""
        사용 알고리즘: {선택알고리즘}

        예측된 아이리스 품종: {품종한글(선택예측)}
        """
    )

    st.write(
        f"꽃잎 길이: {꽃잎길이} cm"
    )

    st.write(
        f"꽃잎 너비: {꽃잎너비} cm"
    )


    # ======================================
    # 13. 세 알고리즘 결과 비교
    # ======================================

    st.divider()

    st.subheader("📊 세 가지 알고리즘 예측 결과 비교")

    비교df = pd.DataFrame(비교자료)

    st.dataframe(
        비교df,
        use_container_width=True,
        hide_index=True
    )

    st.write(
        "동일한 입력값을 세 가지 알고리즘에 적용한 결과입니다."
    )


    # ======================================
    # 14. 품종별 예측 확률
    # ======================================

    st.subheader("📈 알고리즘별 품종 예측 확률")

    확률자료 = []

    전체품종 = sorted(y.unique())

    for 이름 in 모델들.keys():

        모델 = 모델들[이름]

        행 = {
            "알고리즘": 이름
        }

        for 품종, 확률 in zip(
            모델.classes_,
            확률결과[이름]
        ):

            행[품종한글(품종)] = round(
                확률 * 100,
                1
            )

        확률자료.append(행)

    확률df = pd.DataFrame(확률자료)

    st.dataframe(
        확률df,
        use_container_width=True,
        hide_index=True
    )

    그래프df = 확률df.set_index("알고리즘")

    st.bar_chart(그래프df)

    st.caption(
        "※ 예측 확률은 각 알고리즘의 확률 추정치이며 "
        "실제 예측 정확도와는 다릅니다."
    )


    # ======================================
    # 15. KNN 최근접 데이터 5개
    # ======================================

    if 선택알고리즘 == "KNN":

        st.divider()

        st.subheader("🔎 가장 가까운 데이터 5개")

        knn = 모델들["KNN"].named_steps[
            "kneighborsclassifier"
        ]

        scaler = 모델들["KNN"].named_steps[
            "standardscaler"
        ]

        거리, 인덱스 = knn.kneighbors(
            scaler.transform(입력데이터)
        )

        가까운자료 = df.iloc[
            인덱스[0]
        ].copy()

        가까운자료["거리"] = 거리[0]

        가까운자료["품종"] = (
            가까운자료["species"].map(품종한글)
        )

        가까운자료 = 가까운자료[
            [
                "petal_length",
                "petal_width",
                "품종",
                "거리"
            ]
        ]

        가까운자료.columns = [
            "꽃잎 길이",
            "꽃잎 너비",
            "품종",
            "거리"
        ]

        가까운자료.index = range(1, K + 1)

        st.dataframe(
            가까운자료.round(3),
            use_container_width=True
        )


    # ======================================
    # 16. 꽃잎 분포 그래프 (Plotly)
    # ======================================

    st.divider()

    st.subheader("📍 아이리스 꽃잎 분포 그래프")

    fig = go.Figure()

    # 품종별 데이터 표시
    for 품종, 그룹 in df.groupby("species"):

        fig.add_trace(
            go.Scatter(

                x=그룹["petal_length"],

                y=그룹["petal_width"],

                mode="markers",

                name=품종한글(품종),

                marker=dict(
                    size=9,
                    color=품종색상.get(
                        품종,
                        "gray"
                    ),
                    opacity=0.7
                )

            )
        )


    # 사용자 입력값 노란색 별표
    fig.add_trace(
        go.Scatter(

            x=[꽃잎길이],

            y=[꽃잎너비],

            mode="markers+text",

            name="사용자 입력값",

            text=["사용자 입력값"],

            textposition="top right",

            marker=dict(
                size=25,
                color="yellow",
                symbol="star",
                line=dict(
                    color="black",
                    width=1.5
                )
            )

        )
    )


    fig.update_layout(

        title="아이리스 품종별 꽃잎 분포",

        xaxis_title="꽃잎 길이 (cm)",

        yaxis_title="꽃잎 너비 (cm)",

        legend_title="아이리스 품종",

        template="plotly_white",

        height=600,

        font=dict(
            family="sans-serif",
            size=14
        )

    )

    fig.update_xaxes(
        showgrid=True
    )

    fig.update_yaxes(
        showgrid=True
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # ======================================
    # 17. 알고리즘 작동 원리
    # ======================================

    st.divider()

    st.header("🧠 세 가지 분류 알고리즘 작동 원리")


    # --------------------------------------
    # KNN
    # --------------------------------------

    with st.expander(
        "1. KNN (K-최근접 이웃)",
        expanded=True
    ):

        st.markdown(
            """
            **핵심 원리: 나와 가장 비슷한 이웃을 찾는 방법**

            ① 사용자가 꽃잎 길이와 너비를 입력합니다.

            ② 기존 아이리스 데이터와 입력값 사이의 거리를 계산합니다.

            ③ 거리가 가장 가까운 데이터 5개를 찾습니다.

            ④ 가까운 데이터 5개의 품종을 확인합니다.

            ⑤ 가장 많이 등장한 품종을 최종 결과로 선택합니다.
            """
        )

        st.info(
            """
            **쉬운 예시**

            가까운 꽃 5개의 품종이 다음과 같다고 가정해 봅시다.

            - 세토사: 1개
            - 버시컬러: 4개
            - 버지니카: 0개

            가장 많은 표를 받은 버시컬러로 예측합니다.
            """
        )

        st.write(
            "💡 비유: 주변에서 나와 가장 비슷한 친구 5명에게 "
            "의견을 물어보고 가장 많은 의견을 따르는 방법입니다."
        )


    # --------------------------------------
    # Decision Tree
    # --------------------------------------

    with st.expander(
        "2. 의사결정나무 (Decision Tree)"
    ):

        st.markdown(
            """
            **핵심 원리: 여러 가지 질문과 조건을 따라가는 방법**

            ① 꽃잎 길이와 너비를 확인합니다.

            ② 꽃잎 길이가 특정 기준보다 작은지 큰지 확인합니다.

            ③ 조건에 따라 다음 질문으로 이동합니다.

            ④ 여러 가지 조건을 차례대로 확인합니다.

            ⑤ 마지막 단계에 도달하면 품종을 예측합니다.
            """
        )

        st.info(
            """
            **쉬운 예시**

            - 꽃잎 길이가 짧은가요?
            - 꽃잎 너비가 작은가요?
            - 조건에 따라 다음 단계로 이동합니다.
            - 마지막에 해당하는 품종을 예측합니다.
            """
        )

        st.write(
            "💡 비유: 스무고개처럼 질문에 답하면서 "
            "정답을 찾아가는 방법입니다."
        )


    # --------------------------------------
    # Random Forest
    # --------------------------------------

    with st.expander(
        "3. 랜덤 포레스트 (Random Forest)"
    ):

        st.markdown(
            """
            **핵심 원리: 여러 의사결정나무의 의견을 모으는 방법**

            ① 여러 개의 학습용 데이터로 의사결정나무를 만듭니다.

            ② 각 의사결정나무가 독립적으로 품종을 예측합니다.

            ③ 모든 나무의 예측 결과를 모읍니다.

            ④ 각 품종이 받은 표를 계산합니다.

            ⑤ 가장 많은 표를 받은 품종을 최종 결과로 선택합니다.
            """
        )

        st.info(
            """
            **쉬운 예시**

            의사결정나무 100개가 다음과 같이 투표했다고 가정해 봅시다.

            - 세토사: 10표
            - 버시컬러: 75표
            - 버지니카: 15표

            가장 많은 표를 받은 버시컬러로 예측합니다.
            """
        )

        st.write(
            "💡 비유: 여러 명의 전문가에게 의견을 물어보고 "
            "가장 많은 사람이 선택한 답을 결정하는 방법입니다."
        )


    # ======================================
    # 18. 알고리즘 특징 비교
    # ======================================

    st.divider()

    st.header("📚 알고리즘 특징 비교")

    정리자료 = pd.DataFrame({

        "알고리즘": [
            "KNN",
            "의사결정나무",
            "랜덤 포레스트"
        ],

        "작동 원리": [
            "가까운 데이터 비교",
            "조건에 따른 분류",
            "여러 나무의 투표"
        ],

        "주요 특징": [
            "주변 데이터 5개 활용",
            "질문과 조건으로 판단",
            "의사결정나무 100개 활용"
        ]

    })

    st.dataframe(
        정리자료,
        use_container_width=True,
        hide_index=True
    )


    # ======================================
    # 19. 최종 결과
    # ======================================

    st.divider()

    st.success(
        f"선택한 알고리즘: {선택알고리즘}\n\n"
        f"최종 예측 품종: {품종한글(선택예측)}"
    )

    st.caption(
        "※ 세 알고리즘은 동일한 학습 데이터를 사용합니다. "
        "예측 확률은 실제 정확도와 다릅니다."
    )


else:

    st.info(
        "꽃잎 길이와 너비를 입력하고 "
        "알고리즘을 선택한 다음 예측 버튼을 눌러 주세요."
    )


# ==========================================
# 20. 하단 안내
# ==========================================

st.divider()

st.caption(
    "아이리스 인공지능 품종 예측 프로그램 | "
    "판다스 + 사이킷런 + Plotly + Streamlit"
)
