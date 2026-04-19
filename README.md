# 🚀 Sentiment Analysis with End-to-End MLOps Pipeline

## 📌 Overview
This project implements a **production-ready end-to-end MLOps pipeline** for sentiment analysis. It automates the complete machine learning lifecycle including data preprocessing, feature engineering, model training, evaluation, tracking, deployment, and CI/CD integration.

The system is designed to be **scalable, reproducible, and deployment-ready**, leveraging modern MLOps tools and cloud infrastructure.

---

## 🎯 Objective
- Build a robust sentiment analysis model
- Compare multiple ML models and select the best-performing one
- Implement a full MLOps pipeline for automation and reproducibility
- Deploy the model using Docker and AWS infrastructure

---

## 📊 Dataset & Preprocessing
- Text data cleaned using:
  - URL removal
  - Stopword removal
  - Punctuation removal
  - Lemmatization
- Feature extraction using **TF-IDF Vectorization**

---

## 🤖 Models Used
We trained and compared the following models:

- Logistic Regression ✅ (Selected)
- Support Vector Machine (SVM)

---

## 📈 Model Performance

| Model                  | Accuracy | Precision | Recall | F1 Score |
|------------------------|----------|----------|--------|----------|
| Logistic Regression    | **89.39%** | 0.88 | 0.90 | 0.895 |
| SVM                    | 88.26% | 0.87 | 0.88 | 0.883 |

👉 Logistic Regression was selected as the final production model.

---

## ⚙️ MLOps Pipeline

### 🔹 Data Pipeline (DVC)
- Data preprocessing
- Feature engineering (TF-IDF)
- Model training
- Model evaluation

👉 Fully reproducible pipeline using **DVC**

---

### 🔹 Experiment Tracking (MLflow + DagsHub)
- Track metrics, parameters, and models
- Version control for experiments
- Model registry with staging and production lifecycle

---

### 🔹 Model Lifecycle Management
- Model registered in MLflow
- Promoted from **Staging → Production**
- Automated validation before promotion

---

### 🔹 CI/CD Pipeline (GitHub Actions)
Automated pipeline includes:
- DVC pipeline execution (`dvc repro`)
- Model testing
- Flask API testing
- Model promotion
- Docker build
- Push to AWS ECR

---

## 🌐 API Deployment

### 🔹 Flask Application
- REST API for sentiment prediction
- Consistent preprocessing pipeline
- Input → Text → Prediction

---

### 🔹 Dockerization
- Application containerized using Docker
- Lightweight and portable deployment

---

### 🔹 AWS Integration
- Docker image pushed to **AWS Elastic Container Registry (ECR)**
- Designed for deployment on **AWS EKS (Kubernetes)**

---

## 🔄 Local vs Production Setup

The project supports two environments:

### 🧪 Local Development
- Loads model from local `.pkl` files
- Useful for testing and debugging

### 🚀 Production Deployment
- Loads model from **MLflow Production stage**
- Ensures latest validated model is used
- Integrated with DagsHub for remote tracking

---

## 🧠 Tech Stack

- **Machine Learning:** Scikit-learn  
- **Data Pipeline:** DVC  
- **Experiment Tracking:** MLflow + DagsHub  
- **Backend:** Flask  
- **CI/CD:** GitHub Actions  
- **Containerization:** Docker  
- **Cloud:** AWS (ECR, EKS)  

---
