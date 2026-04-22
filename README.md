# 🔮 Telco Customer Churn Prediction

Welcome to the **Telco Customer Churn Predictor**, an end-to-end machine learning project designed to identify customers at risk of leaving a telecommunications service. By understanding these churn drivers, businesses can proactively deploy retention strategies to reduce revenue leakage.

![Live Demo Status](https://img.shields.io/badge/Status-Live_on_Streamlit-success?style=for-the-badge)

## 🎯 The Problem

Customer acquisition is inherently more expensive than customer retention. In the highly competitive telecommunications sector, retaining high-value accounts is critical. This project tackles the problem by analyzing a dataset of **7,043 customers** across **19 distinct features**—ranging from demographics to monthly charges and tech support options.

## 🧠 Model Architecture & Insights

Rather than relying on basic statistical methods, we utilized an **XGBoost Classifier** wrapped in a robust Scikit-learn `Pipeline`. The data undergoes rigorous preprocessing:
- Missing numerical values (like `TotalCharges`) are handled via median imputation and scaled using `StandardScaler`.
- Categorical features (such as `InternetService` and `PaymentMethod`) are transformed via `OneHotEncoder`.

### Performance Metrics
Our model was evaluated on a 20% holdout test set (approx. 1,400 customers) and achieved the following baseline results without heavy hyperparameter tuning:
- **Accuracy:** `79.03%`
- **Precision:** `62.70%` 
- **Recall:** `52.14%`
- **F1 Score:** `56.93%`

*Insight:* While accuracy is strong, the recall score indicates that the model is conservative in flagging churners. In a business context where the cost of a false positive (offering a small discount to someone who wasn't going to leave) is lower than a false negative (losing a high-value customer), future iterations should optimize the decision threshold to improve recall.

## 💻 Tech Stack

This project was built to be entirely open-source and deployable without ongoing cloud costs:
- **Data & Modeling:** Pandas, Scikit-learn, XGBoost
- **Experiment Tracking:** MLflow
- **Frontend UI:** Streamlit (Custom CSS for a premium, responsive design)
- **Deployment:** Streamlit Community Cloud (100% Free)

## 🚀 Running the Project Locally

If you'd like to run the dashboard on your own machine, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/arpitsingh6070/telco-churn-project.git
   cd telco-churn-project
   ```

2. **Set up your environment:**
   Create a virtual environment and install the required dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Launch the Dashboard:**
   Our Streamlit app is configured to load the serialized model (`.pkl`) natively, bypassing the need for a separate backend API.
   ```bash
   streamlit run frontend/app.py
   ```
   The dashboard will be available at `http://localhost:8501`.

---
*Built with ❤️ to demonstrate production-grade ML engineering without the production-grade costs.*
