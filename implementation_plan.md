# End-to-End Telco Customer Churn ML Project

This document outlines the implementation plan to develop and deploy an end-to-end machine learning project for predicting Telco Customer Churn. The project takes inspiration from the referenced GitHub repository but focuses on a completely custom implementation deployed on **100% free-of-cost** cloud resources, giving you full control over the stack.

## Goal
Build a production-ready machine learning pipeline, a RESTful API for inference, and a beautiful, modern web interface. Unlike the original repository which relies on paid AWS services (ECS, ALB), we will use modern free-tier services.



## Proposed Architecture (100% Free)

1. **Model Training & Tracking**:
   - **Tools:** Python, Pandas, Scikit-learn / XGBoost
   - **Tracking:** MLflow (using DagsHub for a free remote tracking server)
   
2. **Backend Inference API**:
   - **Tools:** FastAPI, Uvicorn
   - **Hosting:** [Render](https://render.com/) (Free Tier Web Service) or [Hugging Face Spaces](https://huggingface.co/spaces) (Docker space)

3. **Frontend UI**:
   - **Tools:** React/Vite with a modern, premium design (glassmorphism, modern typography).
   - **Hosting:** [Vercel](https://vercel.com/) (Free Tier) or [Netlify](https://www.netlify.com/).

4. **CI/CD Pipeline**:
   - **Tools:** GitHub Actions to automatically deploy the backend and frontend whenever you push code.

---

## Proposed Implementation Phases

### Phase 1: Model Development (Local)
We will create scripts to process the dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`) and train the model.
#### [NEW] `src/data_preprocessing.py`
#### [NEW] `src/train.py`
#### [NEW] `requirements.txt`

### Phase 2: Backend API Development
We will wrap the trained model in a fast, robust REST API.
#### [NEW] `api/main.py`
#### [NEW] `api/Dockerfile` (if deploying to HF spaces) or `render.yaml`

### Phase 3: Premium Frontend UI Development
We will build a beautiful web interface to interact with the model.
#### [NEW] `frontend/` (React/Vite project setup)
#### [NEW] `frontend/src/App.jsx` (and components)

### Phase 4: CI/CD & Deployment
We will set up the GitHub Actions workflows and deploy the services.
#### [NEW] `.github/workflows/deploy.yml`

## Verification Plan

### Automated Tests
- We will write basic tests (`pytest`) for the FastAPI endpoints to ensure the model responds correctly to valid and invalid inputs.

### Manual Verification
- Verify the model tracks experiments successfully to the MLflow dashboard.
- Test the deployed FastAPI Swagger UI.
- Test the live Vercel frontend app to ensure it can successfully hit the deployed backend API and show churn predictions.
