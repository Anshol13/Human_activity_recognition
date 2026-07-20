import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt

# ----------------------------------------------------
# PAGE CONFIGURATION
# ----------------------------------------------------

st.set_page_config(
    page_title="Human Activity Recognition",
    page_icon="🏃",
    layout="wide"
)

# ----------------------------------------------------
# CUSTOM CSS
# ----------------------------------------------------

st.markdown("""
<style>

.main{
    padding-top:2rem;
}

.block-container{
    padding-top:1rem;
    padding-bottom:2rem;
}

.metric-card{
    background:#1f2937;
    padding:18px;
    border-radius:12px;
    text-align:center;
    border:1px solid #374151;
}

.metric-value{
    font-size:34px;
    font-weight:bold;
    color:#4ade80;
}

.metric-title{
    color:white;
    font-size:18px;
}

.hero{
    background:linear-gradient(90deg,#2563eb,#0f766e);
    padding:30px;
    border-radius:15px;
    color:white;
    margin-bottom:25px;
}

.small-text{
    color:#d1d5db;
    font-size:18px;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# LOAD MODEL
# ----------------------------------------------------

@st.cache_resource
def load_model():

    model = joblib.load("models/random_forest.pkl")
    scaler = joblib.load("models/scaler.pkl")
    encoder = joblib.load("models/label_encoder.pkl")

    return model, scaler, encoder

model, scaler, encoder = load_model()

# ----------------------------------------------------
# SIDEBAR
# ----------------------------------------------------

st.sidebar.image("https://img.icons8.com/color/96/running.png", width=80)

st.sidebar.markdown("# 🏃 HAR")

st.sidebar.caption(
    "Human Activity Recognition using Machine Learning"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📊 Dashboard",
        "🤖 Predict Activity",
        "ℹ About"
    ]
)

# ----------------------------------------------------
# HOME PAGE
# ----------------------------------------------------

if page == "🏠 Home":

    st.markdown("""
    <div class='hero'>
        <h1>🏃 Human Activity Recognition</h1>
        <h4>
        Random Forest Classifier using the WISDM Activity Recognition Dataset
        </h4>
        <p class='small-text'>
        Human Activity Recognition using Machine Learning on the
        WISDM Smartphone Sensor Dataset.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="🎯 Accuracy",
            value="88.07%"
        )

    with col2:
        st.metric(
            label="📊 Features",
            value="43"
        )

    with col3:
        st.metric(
            label="🚶 Activities",
            value="6"
        )

    with col4:
        st.metric(
            label="🤖 Best Model",
            value="Random Forest"
        )

    st.divider()

    left, right = st.columns([2,1])

    with left:

        st.subheader("📖 Project Overview")

        st.write("""
This application classifies six human activities using data collected
from smartphone accelerometer sensors.

The project was built using the **official WISDM dataset**
and compares multiple machine learning algorithms.

The Random Forest classifier achieved the highest accuracy
and was selected as the final prediction model.
""")

        st.success("✅ Best Performing Model: Random Forest")

    with right:

        st.subheader("🏃 Activities")

        st.info("""
- 🚶 Walking

- 🏃 Jogging

- ⬆ Upstairs

- ⬇ Downstairs

- 🪑 Sitting

- 🧍 Standing
""")

    st.divider()

    st.subheader("🏆 Model Comparison")

    comparison = pd.DataFrame({

        "Model":[
            "🥇 Random Forest",
            "🥈 Decision Tree",
            "🥉 KNN"
        ],

        "Accuracy":[
            "88.07%",
            "83.92%",
            "76.14%"
        ]

    })

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )

# ----------------------------------------------------
# DASHBOARD
# ----------------------------------------------------

elif page == "📊 Dashboard":

    st.title("📊 Model Dashboard")

    st.write(
        "Visual summary of the trained Random Forest model and dataset."
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("📈 Activity Distribution")

        st.image(
            "images/activity_distribution.png",
            use_container_width=True
        )

    with col2:

        st.subheader("📊 Model Comparison")

        st.image(
            "images/model_accuracy.png",
            use_container_width=True
        )

    st.divider()

    col3, col4 = st.columns(2)

    with col3:

        st.subheader("🔥 Confusion Matrix")

        st.image(
            "images/confusion_matrix.png",
            use_container_width=True
        )

    with col4:

        st.subheader("⭐ Feature Importance")

        st.image(
            "images/feature_importance.png",
            use_container_width=True
        )

    st.divider()

    st.success(
        "Random Forest achieved the highest accuracy (88.07%) and was selected as the final model."
    )
    st.subheader("🏆 Model Performance")

    performance = pd.DataFrame(
        {
            "Model": [
                "Random Forest",
                "Decision Tree",
                "KNN",
            ],
            "Accuracy": [
                88.07,
                83.92,
                76.14,
            ],
        }
    )

    st.bar_chart(
        performance.set_index("Model")
    )

# ----------------------------------------------------
# PREDICT ACTIVITY
# ----------------------------------------------------

elif page == "🤖 Predict Activity":

    st.title("🤖 Activity Prediction")

    st.markdown("""
Upload a CSV file containing the same features used during model training.

The application automatically:

- Removes the **class** column (if present)
- Removes **UNIQUE_ID** and **user** columns
- Predicts the activity using the trained Random Forest model
- Displays the predicted activities
- Allows you to download the predictions
""")

    sample_file = Path("sample_input.csv")

    if sample_file.exists():

        with open(sample_file, "rb") as f:

            st.download_button(
                label="📥 Download Sample CSV",
                data=f,
                file_name="sample_input.csv",
                mime="text/csv",
                use_container_width=True
            )

    st.write("")
    uploaded_file = st.file_uploader(
        "📤 Upload CSV File",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            # ----------------------------
            # Read CSV
            # ----------------------------
            data = pd.read_csv(uploaded_file)

            st.success("✅ Dataset uploaded successfully!")

            st.subheader("Dataset Preview")

            st.dataframe(
                data.head(),
                use_container_width=True
            )

            col1, col2 = st.columns(2)

            col1.metric("Rows", data.shape[0])
            col2.metric("Columns", data.shape[1])

            st.divider()

            # ----------------------------
            # Predict Button
            # ----------------------------

            if st.button("🚀 Predict Activities", use_container_width=True):

                with st.spinner("Running Random Forest Model..."):

                    prediction_data = data.copy()

                    actual_labels = None

                    # Store actual labels if available
                    if "class" in prediction_data.columns:

                        actual_labels = prediction_data["class"]

                        prediction_data = prediction_data.drop(
                            columns=["class"]
                        )
                    # Remove unnecessary columns
                    prediction_data = prediction_data.drop(
                        columns=["UNIQUE_ID", "user"],
                        errors="ignore"
                    )
                    # ----------------------------------------------------
                    # VALIDATE INPUT DATASET
                    # ----------------------------------------------------

                    # Expected feature columns
                    expected_columns = [
                        'X0','X1','X2','X3','X4','X5','X6','X7','X8','X9',
                        'Y0','Y1','Y2','Y3','Y4','Y5','Y6','Y7','Y8','Y9',
                        'Z0','Z1','Z2','Z3','Z4','Z5','Z6','Z7','Z8','Z9',
                        'XAVG','YAVG','ZAVG',
                        'XPEAK','YPEAK','ZPEAK',
                        'XABSOLDEV','YABSOLDEV','ZABSOLDEV',
                        'XSTANDDEV','YSTANDDEV','ZSTANDDEV',
                        'RESULTANT'
                    ]

                    uploaded_columns = prediction_data.columns.tolist()

                    missing_columns = [
                        col for col in expected_columns
                        if col not in uploaded_columns
                    ]

                    unexpected_columns = [
                        col for col in uploaded_columns
                        if col not in expected_columns
                    ]

                    if missing_columns or unexpected_columns:

                        st.error("❌ Invalid Dataset")

                        if missing_columns:
                            st.write("### Missing Columns")
                            st.write(missing_columns)

                        if unexpected_columns:
                            st.write("### Unexpected Columns")
                            st.write(unexpected_columns)

                        st.stop()

                    # Reorder columns to match training
                    prediction_data = prediction_data[expected_columns]


                    # ----------------------------
                    # Prediction
                    # ----------------------------

                    prediction = model.predict(prediction_data)

                    predicted_labels = encoder.inverse_transform(prediction)

                    result = data.copy()

                    result["Predicted Activity"] = predicted_labels

                    st.success("✅ Prediction completed successfully!")

                    st.subheader("Prediction Results")

                    preview = result.copy()

                    if "class" in preview.columns:
                        preview = preview[
                            ["class", "Predicted Activity"]
                        ]

                    st.dataframe(preview)
                    
                    # ----------------------------------------------------
                    # MODEL EVALUATION
                    # ----------------------------------------------------

                    if actual_labels is not None:

                        st.divider()

                        st.header("📊 Model Evaluation")

                        accuracy = accuracy_score(
                            actual_labels,
                            predicted_labels
                        )

                        col1, col2, col3 = st.columns(3)

                        col1.metric(
                            "Accuracy",
                            f"{accuracy*100:.2f}%"
                        )

                        col2.metric(
                            "Samples",
                            len(actual_labels)
                        )

                        col3.metric(
                            "Classes",
                            len(encoder.classes_)
                        )

                        st.subheader("Classification Report")

                        report = classification_report(
                            actual_labels,
                            predicted_labels,
                            output_dict=True
                        )

                        report_df = pd.DataFrame(report).transpose()

                        report_df = report_df.round(3)

                        st.dataframe(
                            report_df,
                            use_container_width=True
                        )

                        st.subheader("Confusion Matrix")

                        cm = confusion_matrix(
                        actual_labels,
                        predicted_labels,
                        labels=encoder.classes_
                        )

                        fig, ax = plt.subplots(figsize=(7,6))

                        disp = ConfusionMatrixDisplay(
                            confusion_matrix=cm,
                            display_labels=encoder.classes_
                        )
                        disp.plot(
                            cmap="Blues",
                            ax=ax,
                            colorbar=False
                        )
                        st.pyplot(fig)

                        st.subheader("📈 Predicted Activity Distribution")

                        prediction_counts = (
                            pd.Series(predicted_labels)
                            .value_counts()
                            )
                        
                        st.bar_chart(prediction_counts)

                    csv = result.to_csv(index=False).encode("utf-8")

                    st.download_button(
                        label="📥 Download Prediction Results",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

        except Exception as e:

            st.error("❌ Prediction Failed")

            st.error(str(e))

            st.info("""
Possible reasons:

• The uploaded CSV is not the transformed WISDM dataset.

• The dataset columns are incorrect.

• Required columns are missing.

• The uploaded file contains invalid values.
""")

# ----------------------------------------------------
# ABOUT PAGE
# ----------------------------------------------------

elif page == "ℹ About":

    st.title("ℹ About This Project")

    st.markdown("""
    ## 🏃 Human Activity Recognition using Machine Learning

    This project was developed to classify human activities using
    smartphone sensor data from the official **WISDM Activity Recognition Dataset**.

    A comparative study was performed using multiple machine learning algorithms,
    and the **Random Forest Classifier** achieved the highest accuracy.
    """)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("🛠 Technologies Used")

        st.markdown("""
- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Joblib
        """)

    with col2:

        st.subheader("📚 Dataset")

        st.markdown("""
**WISDM Activity Recognition Dataset**

Activities:

- 🚶 Walking
- 🏃 Jogging
- ⬆ Upstairs
- ⬇ Downstairs
- 🪑 Sitting
- 🧍 Standing
        """)

    st.divider()

    st.subheader("⚙ Machine Learning Pipeline")

    st.markdown("""
1. Data Cleaning

2. Feature Engineering

3. Exploratory Data Analysis

4. Model Training

5. Model Comparison

6. Random Forest Selection

7. Model Deployment using Streamlit
""")

    st.divider()

    st.subheader("🌍 Applications")

    st.info("""
🏥 Healthcare Monitoring

⌚ Wearable Devices

🏃 Fitness Tracking

🤖 Smart Environments

🧓 Elderly Activity Monitoring
""")

    st.divider()

    st.success("🎯 Final Model Accuracy : 88.07%")

    st.markdown("---")

    st.markdown(
        """
### 👨‍💻 Developer

**Anshol**

biotechnology | Machine Learning | Data Science
"""
    )

# ----------------------------------------------------
# FOOTER
# ----------------------------------------------------

st.divider()

st.markdown(
    """
<div style='text-align:center; color:gray;'>

Developed by <b>Anshol</b>

Human Activity Recognition using Machine Learning

Powered by Python • Streamlit • Scikit-learn • WISDM Dataset

</div>
""",
unsafe_allow_html=True
)