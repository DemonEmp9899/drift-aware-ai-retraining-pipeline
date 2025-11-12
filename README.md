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

======================================================================
                 🧠 HOW TO RUN: DRIFT-AWARE AI RETRAINING PIPELINE
======================================================================

This guide explains how to set up, configure, and run the project successfully.

----------------------------------------------------------------------
🔧 STEP 1: INSTALL DEPENDENCIES
----------------------------------------------------------------------

1. Make sure you have Python 3.10 or higher installed.

2. Open your terminal inside the project folder:
   M:\Drift aware AI retraining pipeline

3. Install all dependencies using:
   > pip install -r requirements.txt

----------------------------------------------------------------------
🧾 STEP 2: CREATE .ENV FILE
----------------------------------------------------------------------

1. In your project root directory, create a new file named:
   > .env

2. Open the file and add the following environment variables:

   # Supabase
   SUPABASE_URL=your_supabase_url
   SUPABASE_KEY=your_supabase_anon_key

   # OpenRouter (for AI APIs)
   OPENROUTER_API_KEY=your_openrouter_key
   # Configuration
   EMBEDDING_MODEL=openai/text-embedding-ada-002
   CHAT_MODEL=openai/gpt-3.5-turbo

   # Drift thresholds
   EMBEDDING_DRIFT_THRESHOLD=0.15
   REFUSAL_RATE_THRESHOLD=0.05
   TOXICITY_RATE_THRESHOLD=0.02
   ACCURACY_DROP_THRESHOLD=0.05

   # Monitoring
   PROMETHEUS_PORT=8000
   

💡 Tip:
   - Get SUPABASE_URL and SUPABASE_KEY from your Supabase project dashboard.
   - OPENROUTER_API_KEY is only needed if you’re using API-based embedding generation.

----------------------------------------------------------------------
🧱 STEP 3: SET UP THE DATABASE
----------------------------------------------------------------------

1. Go to your Supabase project.
2. Open the SQL Editor.
3. Copy the contents of `setup_database.sql` from your project folder.
4. Paste it into the SQL editor and run it.

✅ This creates the required tables:
   - embeddings_log
   - interaction_log
   - drift_events
   - model_versions

----------------------------------------------------------------------
🧪 STEP 4: GENERATE TEST DATA (OPTIONAL)
----------------------------------------------------------------------

You can simulate user interactions and drift for testing.

To run full simulation:
   > python scripts/simulate.py

To run a quick test simulation (less data, faster):
   > python scripts/simulate.py --quick

This will create mock interactions, drift events, and embedding logs
in your Supabase database automatically.

----------------------------------------------------------------------
⚙️ STEP 5: RUN THE DRIFT PIPELINE
----------------------------------------------------------------------

To run the full drift detection and retraining pipeline:
   > python scripts/run_pipeline.py

✅ This performs:
   - Embedding Drift Detection
   - Behavior Drift Detection (toxicity/refusal)
   - Accuracy Monitoring
   - Automatic response actions (e.g., retrain, reindex, update prompt)

If drift is detected, the system will log it in Supabase
and simulate corrective actions (e.g., reindexing, fine-tuning, etc.).

----------------------------------------------------------------------
📊 STEP 6: RUN METRICS SERVER (FOR GRAFANA)
----------------------------------------------------------------------

Start the Prometheus metrics server:
   > python src/metrics.py

This will start a local server at:
   👉 http://localhost:8000/metrics

Prometheus can scrape metrics from this endpoint, and Grafana
can visualize them in real-time.

----------------------------------------------------------------------
📈 STEP 7: VISUALIZE WITH GRAFANA
----------------------------------------------------------------------

1. Start Grafana and Prometheus on your system.
2. Add Prometheus as a data source (URL: http://localhost:9090)
3. Import the dashboard file from:
   > dashboard/grafana_dashboard.json

✅ You’ll now see live metrics such as:
   - Drift Scores
   - Refusal/Toxicity Rates
   - Accuracy Trends
   - Retraining Events

----------------------------------------------------------------------
🎉 STEP 8: OPTIONAL - CONTINUOUS MONITORING
----------------------------------------------------------------------

To keep the system running periodically (e.g., every 60 minutes):
   > python scripts/run_pipeline.py --continuous 60

💡 You can stop it anytime using:
   Ctrl + C

----------------------------------------------------------------------
✅ STEP 9: COMPLETION CHECK
----------------------------------------------------------------------

If everything is working, you’ll see console output like this:

======================================================================
DRIFT-AWARE RETRAINING PIPELINE
======================================================================
✓ DriftMonitor initialized
🔍 Checking Embedding Drift...
✓ No drift. Score: 0.05 <= 0.15
⚠️ BEHAVIOR DRIFT DETECTED! High toxicity (2.68%)
✅ PIPELINE COMPLETED SUCCESSFULLY
Drifts detected: behavior
Actions taken: update_system_prompt
======================================================================

----------------------------------------------------------------------
💡 OPTIONAL: GIT COMMANDS (IF USING GITHUB)
----------------------------------------------------------------------

To push your project safely to GitHub:

> git init
> git add .
> git commit -m "Initial commit: Drift Aware AI Retraining Pipeline"
> git branch -M main
> git remote add origin https://github.com/<your-username>/<repo-name>.git
> git push -u origin main

Make sure your `.gitignore` file excludes:
   .env
   logs/
   data/
   __pycache__/
   .vscode/

----------------------------------------------------------------------
📧 AUTHOR INFORMATION
----------------------------------------------------------------------

Developed by: Rudra Pratap Tomer
Email: rudratomer.te@gmail.com
Description: Automated Drift Detection & AI Retraining Pipeline
Version: 1.0.0

======================================================================
✨ AI models may drift, but this pipeline never sleeps.
======================================================================
