
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


# ==========================================
# 1. PAGE SETTINGS
# ==========================================

st.set_page_config(
    page_title="Iris AI Classifier",
    page_icon="🌸",
    layout="wide"
)

st.title("🌸 Iris Flower AI Classification")
st.write(
    "Compare three machine learning algorithms "
    "to predict iris flower species."
)

st.divider()


# ==========================================
# 2. LOAD CSV FILE
# ==========================================

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

    raise ValueError("Unable to read CSV file.")


try:
    df = load_data(str(file_path))

except Exception as e:
    st.error(f"File loading error: {e}")
    st.stop()


# ==========================================
# 3. DATA PREPROCESSING
# ==========================================

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


# ==========================================
# 4. SPECIES NAME
# ==========================================

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


# ==========================================
# 5. INPUT RANGE
# ==========================================

min_length = float(X["petal_length"].min())
max_length = float(X["petal_length"].max())

min_width = float(X["petal_width"].min())
max_width = float(X["petal_width"].max())


# ==========================================
# 6. CREATE THREE MODELS
# ==========================================

models = {

    "KNN": make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=5)
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=4,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=4,
        random_state=42
    )

}


# ==========================================
# 7. TRAIN ALL MODELS
# ==========================================

for model in models.values():
    model.fit(X, y)


# ==========================================
# 8. DATA INFORMATION
# ==========================================

st.subheader("Dataset Information")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Samples",
    len(df)
)

col2.metric(
    "Number of Species",
    y.nunique()
)

col3.metric(
    "Algorithms",
    "3"
)

st.divider()


# ==========================================
# 9. USER INPUT
# ==========================================

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


# ==========================================
# 10. SELECT ALGORITHM
# ==========================================

st.divider()

st.subheader("🤖 Select Machine Learning Algorithm")

selected_algorithm = st.selectbox(
    "Choose an algorithm",
    [
        "KNN",
        "Decision Tree",
        "Random Forest"
    ]
)

algorithm_descriptions = {
    "KNN": "Finds the 5 closest flowers and predicts by voting.",
    "Decision Tree": "Uses a sequence of conditions to classify flowers.",
    "Random Forest": "Combines 100 decision trees to make a prediction."
}

st.info(
    algorithm_descriptions[selected_algorithm]
)


# ==========================================
# 11. PREDICTION BUTTON
# ==========================================

if st.button(
    "Predict and Compare Algorithms",
    use_container_width=True
):

    input_data = pd.DataFrame(
        [[petal_length, petal_width]],
        columns=["petal_length", "petal_width"]
    )

    # --------------------------------------
    # Predict using all three algorithms
    # --------------------------------------

    predictions = {}
    probability_results = {}
    comparison_rows = []

    for name, model in models.items():

        prediction = model.predict(input_data)[0]

        probabilities = model.predict_proba(
            input_data
        )[0]

        predictions[name] = prediction

        probability_results[name] = probabilities

        confidence = max(probabilities) * 100

        comparison_rows.append({
            "Algorithm": name,
            "Predicted Species": get_species_name(prediction),
            "Confidence (%)": round(confidence, 1)
        })


    # ======================================
    # 12. SELECTED ALGORITHM RESULT
    # ======================================

    st.divider()

    st.subheader("Selected Algorithm Result")

    selected_prediction = predictions[selected_algorithm]

    st.success(
        f"""
        Selected Algorithm: {selected_algorithm}

        Predicted Species: {get_species_name(selected_prediction)}
        """
    )

    st.write(
        f"Petal Length: {petal_length} cm | "
        f"Petal Width: {petal_width} cm"
    )


    # ======================================
    # 13. COMPARE THREE ALGORITHMS
    # ======================================

    st.divider()

    st.subheader("📊 Compare Three Algorithms")

    comparison_df = pd.DataFrame(comparison_rows)

    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True
    )

    st.write(
        "All three algorithms predict the same input data. "
        "Compare their predicted species and confidence values."
    )

    # ======================================
    # 14. SPECIES PROBABILITY COMPARISON
    # ======================================

    st.subheader("Species Probability Comparison")

    probability_rows = []

    for name in models.keys():

        classes = models[name].classes_

        row = {
            "Algorithm": name
        }

        for species, probability in zip(
            classes,
            probability_results[name]
        ):

            row[get_species_name(species)] = round(
                probability * 100, 1
            )

        probability_rows.append(row)

    probability_df = pd.DataFrame(probability_rows)

    st.dataframe(
        probability_df,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Note: Confidence values are model probability estimates. "
        "Their interpretation differs by algorithm."
    )


    # ======================================
    # 15. PROBABILITY CHART
    # ======================================

    st.subheader("Prediction Probability Chart")

    chart_df = probability_df.set_index("Algorithm")

    st.bar_chart(chart_df)


    # ======================================
    # 16. SCATTER PLOT
    # ======================================

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

    # User input as yellow star
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


    # ======================================
    # 17. ALGORITHM EXPLANATIONS
    # ======================================

    st.divider()

    st.header("🧠 How Do the Three Algorithms Work?")

    # --------------------------------------
    # KNN
    # --------------------------------------

    st.subheader("1. KNN (K-Nearest Neighbors)")

    st.write(
        """
        KNN predicts a flower by looking at the
        5 most similar flowers in the existing dataset.
        """
    )

    st.markdown(
        """
        **How it works:**

        1. Receive the user's petal length and width.
        2. Calculate the distance between the input and existing flowers.
        3. Find the 5 closest flowers.
        4. Count the species of those 5 flowers.
        5. Predict the species with the most votes.

        **Easy example:**

        Suppose the 5 closest flowers are:

        - Setosa: 1
        - Versicolor: 4
        - Virginica: 0

        The final prediction is Versicolor.
        """
    )

    st.info(
        "KNN is like asking your 5 closest neighbors "
        "which flower looks most similar."
    )

    # --------------------------------------
    # Decision Tree
    # --------------------------------------

    st.subheader("2. Decision Tree")

    st.write(
        """
        A Decision Tree predicts a species by following
        a series of questions and conditions.
        """
    )

    st.markdown(
        """
        **How it works:**

        1. Look at the petal measurements.
        2. Check a condition, such as whether petal length
           is smaller or larger than a certain value.
        3. Follow the appropriate branch.
        4. Continue checking conditions.
        5. Reach a final group and predict its most common species.

        **Easy example:**

        - Is petal length small?
        - Is petal width small?
        - Follow the answers to the next question.
        - Reach the final species prediction.
        """
    )

    st.info(
        "A Decision Tree works like a questionnaire "
        "that asks questions one by one."
    )

    # --------------------------------------
    # Random Forest
    # --------------------------------------

    st.subheader("3. Random Forest")

    st.write(
        """
        Random Forest combines many Decision Trees
        to make one final prediction.
        """
    )

    st.markdown(
        """
        **How it works:**

        1. Create many Decision Trees using different
           samples and feature selections.
        2. Each tree makes its own prediction.
        3. Collect the predictions from all trees.
        4. Count how many trees voted for each species.
        5. Predict the species with the most votes.

        **Easy example:**

        Suppose 100 trees predict:

        - Setosa: 10 votes
        - Versicolor: 75 votes
        - Virginica: 15 votes

        The final prediction is Versicolor.
        """
    )

    st.info(
        "Random Forest is like asking 100 experts "
        "and choosing the answer with the most votes."
    )


    # ======================================
    # 18. FINAL COMPARISON
    # ======================================

    st.divider()

    st.header("📚 Algorithm Comparison Summary")

    summary_df = pd.DataFrame({
        "Algorithm": [
            "KNN",
            "Decision Tree",
            "Random Forest"
        ],
        "Main Idea": [
            "Compare nearby data",
            "Follow decision rules",
            "Combine many trees"
        ],
        "Voting Method": [
            "5 nearest neighbors",
            "Samples in a leaf",
            "100 decision trees"
        ]
    })

    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True
    )

    st.success(
        f"Your selected algorithm is {selected_algorithm}. "
        f"Its prediction is {get_species_name(selected_prediction)}."
    )

else:

    st.info(
        "Enter the flower measurements, select an algorithm, "
        "and click the prediction button."
    )


# ==========================================
# 19. FOOTER
# ==========================================

st.divider()

st.caption(
    "Iris AI Classification | "
    "Pandas + Scikit-learn + Streamlit"
)
