# 🤖 Drift-Aware AI Retraining Pipeline

> A complete automated system that **detects AI model drift**, logs interactions, monitors performance, and **triggers retraining actions** — ensuring your model stays accurate and reliable over time.

---

## 🌍 Overview

As AI systems interact with users over time, their performance can degrade due to **data drift**, **behavior drift**, or **accuracy loss**.  
This project provides a **fully automated drift detection and retraining pipeline** built using **Python**, **Supabase**, **Prometheus**, and **Grafana**.

It continuously monitors model outputs, user feedback, embeddings, and performance metrics — then **detects, logs, and responds to drift** automatically.

---

## 🎯 Objective

The main goal is to:
- 🧠 Detect when an AI model starts performing poorly.
- 🔍 Identify *what kind* of drift is happening (embedding, behavior, or accuracy).
- ⚙️ Automatically trigger retraining or reindexing to recover performance.
- 📊 Provide a live metrics dashboard using Prometheus + Grafana.

---

## 🚀 Features

| Feature | Description |
|----------|-------------|
| 🧩 **Drift Detection** | Detects embedding drift, behavior drift, and accuracy degradation using statistical checks. |
| 🔁 **Automated Retraining Triggers** | Automatically runs reindexing or retraining when drift is detected. |
| 📦 **Data Logging** | Logs embeddings and user interactions in Supabase for analysis. |
| 📊 **Real-time Monitoring** | Exports live metrics via Prometheus for Grafana dashboards. |
| 🧠 **Supabase Backend** | Stores embeddings, feedback, and drift events securely in the cloud. |
| ⚙️ **Simulation Mode** | Generates realistic user interaction data for testing the pipeline. |

---

## 🧠 Architecture Overview

```mermaid
flowchart TD
    A[User Interaction / Simulation] --> B[Data Logger]
    B --> C[Supabase Database]
    C --> D[Drift Monitor]
    D -->|Drift Detected| E[Trigger Actions]
    D -->|No Drift| F[Healthy State]
    E --> G[Retraining / Reindexing]
    G --> C
    D --> H[Prometheus Metrics Exporter]
    H --> I[Grafana Dashboard]
