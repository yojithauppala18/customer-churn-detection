# 📉 Customer Churn Detection – End-to-End Machine Learning Project

## 🌟 What is this project about?
Imagine you run a company and some customers suddenly stop using your service.  
This project builds a machine learning model that predicts **which customers are likely to leave (churn)** so businesses can take action early and retain them.

This repository contains a **complete, end-to-end customer churn prediction system** built using real-world data and industry-standard machine learning practices.

---

## 🎯 Why this project matters
Customer churn directly affects revenue and customer growth.  
This project demonstrates my ability to:
- Understand real business problems
- Work with real-world messy data
- Build clean and scalable ML pipelines
- Evaluate models correctly and responsibly

---

## 🧠 Problem Statement
**Goal:**  
Predict whether a customer will **churn (Yes)** or **stay (No)** based on their personal details, services used, and billing information.

---

## 📊 Dataset Description (Simple Explanation)

### 📁 Dataset Used
**Telco Customer Churn Dataset**

Each row represents **one customer**, and each column describes **something about that customer**.

### 🎯 Target Variable
- **Churn**  
  - `Yes` → Customer left the company  
  - `No` → Customer stayed

### 🧍 Customer Information
- `gender` – Male or Female  
- `SeniorCitizen` – Whether the customer is a senior citizen  
- `Partner` – Has a partner or not  
- `Dependents` – Has dependents or not  

### 📞 Services Used
- `PhoneService`
- `MultipleLines`
- `InternetService`
- `OnlineSecurity`
- `OnlineBackup`
- `DeviceProtection`
- `TechSupport`
- `StreamingTV`
- `StreamingMovies`

These columns help understand **what services make customers stay or leave**.

### 💳 Account & Billing Details
- `tenure` – How long the customer has stayed with the company  
- `Contract` – Month-to-month, one year, or two year  
- `PaperlessBilling`
- `PaymentMethod`
- `MonthlyCharges`
- `TotalCharges`

---
## 📂 Project Structure

```text
customer-churn/
│
├── data/                         # Raw customer data
│   └── Telco Customer Churn.csv
│
├── notebooks/                    # Exploratory Data Analysis
│   └── eda.ipynb
│
├── src/                          # End-to-end ML pipeline
│   ├── data_loader.py            # Load and clean data
│   ├── preprocessing.py         # Feature engineering & preprocessing
│   ├── train.py                  # Model training & evaluation
│   ├── main.py                   # Pipeline execution
│   └── config.py                 # Configurable parameters
│
├── models/                       # Saved trained models
│   └── best_churn_model.pkl
│
└── README.md

```
---

## 🔍 Step 1: Exploratory Data Analysis
Before training models, the data is explored to understand:
- How many customers churn vs stay
- How churn changes with tenure and monthly charges
- Which services are linked to higher churn
- Missing values and data quality issues

📓 Notebook: `notebooks/eda.ipynb`

---

## ⚙️ Step 2: Data Preprocessing
Machine learning models need clean and numerical data.

### 🔢 Numerical Features
- Missing values are filled
- Values are scaled for better learning

### 🏷️ Categorical Features
- Text values are converted into numbers
- Unknown categories are handled safely

All preprocessing is implemented using **Scikit-learn Pipelines**, making the workflow reproducible and production-ready.

---

## 🤖 Step 3: Models Trained
The following models are trained and compared:
- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier

---
## 📊 Step 4: Model Selection & Evaluation

To ensure reliable and fair predictions, multiple machine learning models were trained and evaluated instead of relying on a single algorithm. Each model was chosen for a specific reason.

### 🤖 Why These Models Were Chosen

**1. Logistic Regression**
- Serves as a strong baseline model
- Easy to interpret and explain to business stakeholders
- Performs well for binary classification problems like churn

**2. Random Forest Classifier**
- Captures non-linear relationships in customer behavior
- Handles mixed data types effectively
- Reduces overfitting by averaging multiple decision trees

**3. XGBoost Classifier**
- High-performance gradient boosting algorithm
- Excellent at learning complex patterns in structured data
- Frequently used in real-world churn and risk prediction systems

Using these three models allows comparison between:
- A simple linear model
- An ensemble-based model
- A state-of-the-art boosting model

---

### 📏 Why These Evaluation Methods Were Used

**Stratified K-Fold Cross-Validation**
- Ensures both churned and non-churned customers are evenly distributed across folds
- Prevents biased evaluation caused by class imbalance
- Provides a stable and reliable estimate of model performance

**ROC-AUC Score**
- Measures how well the model separates churned and non-churned customers
- Works well even when classes are imbalanced
- More informative than accuracy for churn prediction problems

This evaluation strategy ensures that the selected model performs consistently across different data splits.

---

## 🏆 Step 5: Best Model Selection & Final Training

After evaluating all models using cross-validation:

- Mean ROC-AUC scores were compared across models
- The model with the highest and most stable performance was selected
- The selected model was retrained on the full dataset
- The final trained model was saved for future predictions

📦 Saved Model:
## 🚀 How to Run the Project
Run the full pipeline using:
```bash
python src/main.py
```
---
## 🧩 Skills Demonstrated
- Machine Learning & Classification
- Data Preprocessing & Feature Engineering
- Cross-Validation & Model Evaluation
- Model Selection & Comparison
- Scikit-learn Pipelines
- Business-Oriented Problem Solving
- Clean & Modular Code Design
---
## 🛠️ Technologies Used
- Python
- pandas
- scikit-learn
- XGBoost
- joblib
---
## ⭐ Final Note

- This project demonstrates my ability to take a real-world business problem, understand the data, build reliable machine learning models, and deliver a scalable solution using industry best practices.
---
## 👤 Author

- Yojitha Uppala
- MS in Business Analytics & Artificial Intelligence
- The University of Texas at Dallas

