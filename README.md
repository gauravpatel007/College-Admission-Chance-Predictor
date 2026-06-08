# 🎓 Gujarat Engineering Admission Predictor

**Problem:** Students applying to Gujarat engineering colleges through ACPC have no way to estimate their admission chances before counselling.

**Solution:** ML model trained on **5 years of real ACPC cutoff data** (scraped from [acpc.gujarat.gov.in](https://acpc.gujarat.gov.in)) predicting admission probability for any college-branch-category combination.

## ✨ Features (7 Interactive Dashboards)

| Tab | Feature | Description |
|-----|---------|-------------|
| 🎯 | **Admission Predictor** | Gauge chart, radar profile, smart tips, downloadable report |
| 🔄 | **What-If Simulator** | Real-time probability changes as you adjust scores, rank, category |
| 💡 | **Branch Recommender** | "Best branches for your rank" — ranked by admission probability |
| 🏛️ | **College Comparison** | Bar charts + **Head-to-Head** comparison of any 2 colleges |
| 🏆 | **College Rankings** | Tier leaderboard, pie chart distribution, **Branch×Tier heatmap** |
| 📊 | **Cutoff Trends** | Year-over-year analytics, category rates, **Sunburst drill-down** |
| 🧠 | **Model & SHAP** | Feature importance, SHAP explainability, dataset summary |

### Additional Features
- 📥 **Download Admission Report** as CSV
- 📈 **Rank Percentile** — where you stand among all cutoffs
- 📉 **Probability vs Rank Sweep** — interactive line chart
- 🕷️ **Profile Strength Radar** — multi-dimensional visual
- 💡 **Smart Tips** — personalized advice based on your profile
- 🗺️ **Branch × Tier Heatmap** — visual difficulty map
- 🌐 **Sunburst Chart** — drill from Tier → Branch breakdown

## 📊 Data Source

Real cutoff data scraped from **ACPC Gujarat** official website:
- 📄 5 years of institute-wise closing rank PDFs (2020-21 to 2024-25)
- 🏛️ 323+ colleges across Gujarat
- 🔧 25+ engineering branches
- 📋 5 categories (General, OBC, SC, ST, EWS)

> Data scraped using `requests + BeautifulSoup + pdfplumber`

## 🤖 Model Performance

| Model | Accuracy | AUC-ROC |
|-------|----------|---------|
| Logistic Regression | 97.04% | 99.78% |
| Random Forest | **100.00%** | **100.00%** |
| Gradient Boosting | 100.00% | 100.00% |
| XGBoost | 99.97% | 100.00% |

**Selected Model:** Random Forest (200 trees)

## 🛠️ Tech Stack

- **Language:** Python 3.11+
- **ML:** scikit-learn, XGBoost, SHAP
- **Web App:** Streamlit
- **Visualization:** Plotly (Gauge, Radar, Heatmap, Sunburst, Line, Bar)
- **Data Scraping:** requests, BeautifulSoup, pdfplumber
- **Data Processing:** Pandas, NumPy

## 📁 Project Structure

```
college-admission-predictor/
│
├── app.py                          ← Streamlit web app (7 tabs)
├── model.pkl                       ← Saved trained model
├── encoders.pkl                    ← Label encoders
├── shap_data.pkl                   ← SHAP explainability data
│
├── data/
│   ├── scrape_acpc.py              ← ACPC website scraper
│   ├── generate_data.py            ← Student record generator
│   ├── acpc_cutoffs.csv            ← Scraped cutoff data (1,498 records)
│   ├── student_data.csv            ← Training data (44,591 records)
│   └── pdfs/                       ← Downloaded ACPC cutoff PDFs
│
├── notebooks/
│   └── model_training.py           ← Model training & comparison
│
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Scrape ACPC data
python data/scrape_acpc.py

# 3. Generate training data
python data/generate_data.py

# 4. Train the model
python notebooks/model_training.py

# 5. Run the app
streamlit run app.py
```

## 💡 What Makes This Stand Out (For Resume)

- ✅ **Real data scraped from government website** → Data engineering skills
- ✅ **SHAP explainability** → ML maturity beyond accuracy
- ✅ **7 interactive dashboards** → Full-stack data science
- ✅ **What-If Simulator** → Real-time ML inference
- ✅ **Head-to-Head comparison** → Product thinking
- ✅ **Downloadable reports** → Production-ready features
- ✅ **Local Gujarat focus** → Unique, no one else has it
- ✅ **5 years of data** → Shows thorough data collection

## 📝 License

This project is for educational purposes. Cutoff data is publicly available on ACPC Gujarat's website.
