
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


# =====================================
# 1. PAGE SETTINGS
# =====================================

st.set_page_config(
    page_title="Iris Flower Prediction",
    page_icon="🌸",
    layout="wide"
)

st.title("🌸 Iris Flower Prediction")

st.write(
    "Enter petal length and width to predict "
    "the iris flower species."
)

st.divider()


# =====================================
# 2. LOAD CSV FILE
# =====================================

current_folder = Path(__file__).resolve().parent

file_path = current_folder / "Iris.csv"

if not file_path.exists():
    file_path = current_folder.parent / "Iris.csv"

if not file_path.exists():
    st.error("Iris.csv file not found.")
    st.stop()


@st.cache_data
def load_data(path):

    encodings = [
        "cp949",
        "utf-8-sig",
        "utf-8",
        "euc-kr"
    ]

    for encoding in encodings:
        try:
            return pd.read_csv(
                path,
                encoding=encoding
            )
        except UnicodeDecodeError:
            continue

    raise ValueError("Unable to read CSV encoding.")


try:
    df = load_data(str(file_path))

except Exception as e:
    st.error(f"File loading error: {e}")
    st.stop()


# =====================================
# 3. DATA PREPROCESSING
# =====================================

df.columns = df.columns.astype(str).str.strip()

column_aliases = {
    "petal_length": [
        "PetalLengthCm",
        "Petal Length",
        "petal_length",
        "꽃잎 길이"
    ],
    "petal_width": [
        "PetalWidthCm",
        "Petal Width",
        "petal_width",
        "꽃잎 너비"
    ],
    "species": [
        "Species",
        "species",
        "품종"
    ]
}

rename_dict = {}

for standard, aliases in column_aliases.items():
    for alias in aliases:
        if alias in df.columns:
            rename_dict[alias] = standard
            break

df = df.rename(columns=rename_dict)

required_columns = [
    "petal_length",
    "petal_width",
    "species"
]

if not all(col in df.columns for col in required_columns):
    st.error("Required CSV columns are missing.")
    st.write("Available columns:", df.columns.tolist())
    st.stop()

df["petal_length"] = pd.to_numeric(
    df["petal_length"],
    errors="coerce"
)

df["petal_width"] = pd.to_numeric(
    df["petal_width"],
    errors="coerce"
)

df = df.dropna(
    subset=required_columns
).copy()

if len(df) < 5:
    st.error("Not enough training data.")
    st.stop()

X = df[["petal_length", "petal_width"]]
y = df["species"].astype(str)


# =====================================
# 4. SPECIES NAME
# =====================================

species_names = {
    "Iris-setosa": "Setosa",
    "Iris-versicolor": "Versicolor",
    "Iris-virginica": "Virginica",
    "setosa": "Setosa",
    "versicolor": "Versicolor",
    "virginica": "Virginica"
}

def get_species_name(name):
    return species_names.get(name, name)


species_colors = {
    "Iris-setosa": "blue",
    "Iris-versicolor": "green",
    "Iris-virginica": "red",
    "setosa": "blue",
    "versicolor": "green",
    "virginica": "red"
}


# =====================================
# 5. INPUT RANGE
# =====================================

min_length = float(X["petal_length"].min())
max_length = float(X["petal_length"].max())

min_width = float(X["petal_width"].min())
max_width = float(X["petal_width"].max())


# =====================================
# 6. TRAIN KNN MODEL
# =====================================

K = 5

model = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=K)
)

model.fit(X, y)


# =====================================
# 7. DATA INFORMATION
# =====================================

st.subheader("Dataset Information")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Samples",
    f"{len(df)}"
)

col2.metric(
    "Petal Length Range",
    f"{min_length} - {max_length} cm"
)

col3.metric(
    "Petal Width Range",
    f"{min_width} - {max_width} cm"
)


# =====================================
# 8. USER INPUT
# =====================================

st.divider()

st.subheader("Enter Flower Measurements")

st.info(
    f"""
    Available Input Ranges

    - Petal Length: {min_length} - {max_length} cm
    - Petal Width: {min_width} - {max_width} cm
    """
)

col1, col2 = st.columns(2)

with col1:

    petal_length = st.number_input(
        "Petal Length (cm)",
        min_value=min_length,
        max_value=max_length,
        value=min(5.0, max_length),
        step=0.1,
        format="%.1f"
    )

with col2:

    petal_width = st.number_input(
        "Petal Width (cm)",
        min_value=min_width,
        max_value=max_width,
        value=min(1.5, max_width),
        step=0.1,
        format="%.1f"
    )


# =====================================
# 9. PREDICTION
# =====================================

if st.button(
    "Predict Iris Species",
    use_container_width=True
):

    input_data = pd.DataFrame(
        [[petal_length, petal_width]],
        columns=["petal_length", "petal_width"]
    )

    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]

    # Find nearest 5 samples
    knn = model.named_steps["kneighborsclassifier"]
    scaler = model.named_steps["standardscaler"]

    scaled_input = scaler.transform(input_data)
    scaled_X = scaler.transform(X)

    distances, indices = knn.kneighbors(
        scaled_input
    )

    nearest_data = df.iloc[indices[0]].copy()

    nearest_data["Distance"] = distances[0]

    nearest_data["Species Name"] = (
        nearest_data["species"].map(get_species_name)
    )

    st.divider()

    # =================================
    # 10. ALGORITHM INFORMATION
    # =================================

    st.subheader("🤖 Machine Learning Algorithm")

    st.success(
        "Algorithm Used: K-Nearest Neighbors (KNN)"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Algorithm", "KNN")
    col2.metric("Nearest Neighbors (K)", K)
    col3.metric("Features Used", "2")

    st.write(
        """
        KNN is a classification algorithm that predicts
        a new sample by comparing it with similar samples
        in the existing dataset.
        """
    )

    # =================================
    # 11. PREDICTION RESULT
    # =================================

    st.divider()

    st.subheader("Prediction Result")

    st.success(
        f"Predicted Species: {get_species_name(prediction)}"
    )

    st.write(
        f"Your Input: Petal Length = {petal_length} cm, "
        f"Petal Width = {petal_width} cm"
    )

    # =================================
    # 12. NEAREST 5 DATA
    # =================================

    st.subheader("🔎 The 5 Nearest Neighbors")

    result_table = nearest_data[
        [
            "petal_length",
            "petal_width",
            "Species Name",
            "Distance"
        ]
    ].copy()

    result_table.columns = [
        "Petal Length (cm)",
        "Petal Width (cm)",
        "Species",
        "Distance"
    ]

    result_table.index = range(1, K + 1)

    st.dataframe(
        result_table.round(3),
        use_container_width=True
    )

    # Count species among neighbors
    neighbor_counts = nearest_data[
        "Species Name"
    ].value_counts()

    st.write("Voting Results")

    for species, count in neighbor_counts.items():

        st.write(
            f"{species}: {count} out of {K} neighbors"
        )

    # =================================
    # 13. PREDICTION PROBABILITY
    # =================================

    st.divider()

    st.subheader("Prediction Probability")

    probability_df = pd.DataFrame({
        "Species": [
            get_species_name(name)
            for name in model.classes_
        ],
        "Probability (%)": probabilities * 100
    })

    st.bar_chart(
        probability_df.set_index("Species"),
        y="Probability (%)"
    )

    # =================================
    # 14. SCATTER PLOT
    # =================================

    st.divider()

    st.subheader("Iris Petal Distribution")

    fig, ax = plt.subplots(figsize=(11, 7))

    for species, group in df.groupby("species"):

        ax.scatter(
            group["petal_length"],
            group["petal_width"],
            color=species_colors.get(species, "gray"),
            label=get_species_name(species),
            alpha=0.65,
            s=65
        )

    # User input
    ax.scatter(
        petal_length,
        petal_width,
        color="yellow",
        edgecolors="black",
        marker="*",
        s=500,
        linewidths=1.5,
        label="Your Input",
        zorder=10
    )

    ax.annotate(
        "Your Input",
        (petal_length, petal_width),
        xytext=(12, 12),
        textcoords="offset points",
        fontsize=11,
        fontweight="bold"
    )

    ax.set_title(
        "Iris Species Petal Distribution",
        fontsize=17
    )

    ax.set_xlabel(
        "Petal Length (cm)",
        fontsize=13
    )

    ax.set_ylabel(
        "Petal Width (cm)",
        fontsize=13
    )

    ax.legend(title="Species")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    # =================================
    # 15. EASY ALGORITHM EXPLANATION
    # =================================

    st.divider()

    st.subheader("🧠 How Does KNN Work?")

    st.write(
        """
        KNN stands for K-Nearest Neighbors.

        It predicts a flower species by finding
        the 5 most similar flowers in the dataset.
        """
    )

    st.markdown(
        """
        ### Step 1. Enter the measurements

        You enter the petal length and petal width
        of a flower.

        ### Step 2. Find similar flowers

        The computer compares your input with
        all flowers in the training dataset.

        ### Step 3. Find the 5 closest flowers

        KNN selects the 5 flowers that have
        the smallest distance from your input.

        ### Step 4. Vote for the species

        The 5 flowers vote for their species.
        The species with the most votes wins.

        ### Step 5. Show the prediction

        The computer displays the species
        with the most votes as the final prediction.
        """
    )

    st.info(
        f"""
        Example:

        If the 5 nearest flowers are:

        - Setosa: 1 flower
        - Versicolor: 4 flowers
        - Virginica: 0 flowers

        The prediction is Versicolor because
        it received the most votes.
        """
    )

    st.caption(
        "Note: KNN uses standardized petal measurements "
        "to calculate distances."
    )

else:

    st.info(
        "Enter the measurements and click "
        "'Predict Iris Species'."
    )


# =====================================
# 16. FOOTER
# =====================================

st.divider()

st.caption(
    "Iris Flower Prediction | "
    "Pandas + Scikit-learn + Streamlit"
)
