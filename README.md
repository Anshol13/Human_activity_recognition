# 🏃 Human Activity Recognition using Machine Learning

A machine learning web application that classifies human activities using smartphone accelerometer sensor data from the WISDM Activity Recognition Dataset.

---

## 🚀 Live Demo

https://humanactivityrecognition-anshol13.streamlit.app/

---

## 📌 Features

- Predicts six human activities using a trained Random Forest model
- Upload CSV files for prediction
- Automatic dataset validation
- Classification report
- Confusion matrix
- Prediction distribution chart
- Download prediction results
- Interactive Streamlit dashboard

---

## 📊 Dataset

- **Dataset:** WISDM Activity Recognition Dataset
- Activities:
  - Walking
  - Jogging
  - Upstairs
  - Downstairs
  - Sitting
  - Standing

---

## 🤖 Machine Learning Models

| Model | Accuracy |
|--------|----------|
| Random Forest | **88.07%** |
| Decision Tree | 83.92% |
| KNN | 76.14% |

Random Forest was selected as the final deployed model.


---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Joblib

---

## 📂 Project Structure

```text
Human_activity_recognition/
│
├── app.py
├── Human_Activity_Recognition.ipynb
├── sample_input.csv
├── requirements.txt
├── README.md
├── models/
├── images/
└── WISDM_ar_v1.1_transformed.arff
```

---

## Results

![Activity Distribution](images/activity_distribution.png)

![Model Accuracy](images/model_accuracy.png)

![Confusion Matrix](images/confusion_matrix.png)

![Feature Importance](images/feature_importance.png)

---

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py

---

## Author

Anshol Prasad