# 🍽️ Amazon Fine Food – Voice of Customer (VoC) Analytics Project  

## 🎯 Project Goal
This project analyzes **Amazon Fine Food Reviews** to extract actionable **Voice of Customer (VoC) insights**.  

The focus is on:  
- **NPS (Net Promoter Score)**  
- **CSAT (Customer Satisfaction)**  
- **CES (Customer Effort Score)**  
- **Sentiment & Topic Modeling**  

to uncover **customer pain points (detractors)** and **delighters (promoters)**, demonstrating how VoC analytics can guide **customer-centric business strategy**.  

---

## 📊 Dataset
- **Source**: [Amazon Fine Food Reviews – Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)  
- **Size**: ~500,000 reviews (~500 MB, not included in repo)  
- **Format**: CSV  

> ⚠️ Note: The raw dataset is too large to store in GitHub.  
> Instead, this repo includes **cleaned and tidy-format CSVs** for direct use in Tableau.  

---

## 🔑 Key Insights (Examples)
- **Overall NPS**: +42  
- **Overall CSAT**: 4.2 / 5  
- **Top Detractor Themes**: packaging issues, stale food, delivery delays.  
- **Top Promoter Themes**: freshness, taste, fast shipping.  
- **Model Drivers**: Logistic Regression highlighted **“fresh”, “delicious” (positive)** vs. **“stale”, “broken”, “refund” (negative)**.  

---

## 🛠️ Methods
### 1. Data Cleaning & Feature Engineering  
- Processed >500K reviews (sampled for efficiency).  
- Built NPS labels (**Promoter / Passive / Detractor**).  
- Engineered features: helpfulness ratio, text length, exclamation count, capitalization ratio, sentiment score, CES keywords.  

### 2. Metrics Framework  
- Computed **NPS & CSAT overall, monthly, and product-level**.  
- Applied **winsorization** to remove outlier bias.  
- Used **minimum thresholds** (30 reviews per product, 100 reviews per month) for reliability.  

### 3. Predictive Modeling  
- Built a **Logistic Regression model** to classify promoters vs. detractors.  
- Identified **key drivers of satisfaction** via top positive/negative features.  

### 4. Topic Modeling (LDA)  
- Extracted recurring **themes from review text**.  
- Grouped into **Pain Points (detractors)** and **Delighters (promoters)**.  

### 5. Visualization  
- Exported tidy CSVs for Tableau.  
- Built **dashboard**:  
  - NPS & CSAT overall  
  - Monthly trends  
  - Top 10 products by NPS  
  - Bottom 10 products by NPS  

---

## 📈 Dashboard
The interactive dashboard (Tableau) is available as a PDF:  

👉 [📊 View Dashboard (PDF)](https://github.com/rgao77/Amazon-FineFood-VoC-Project/blob/main/VoC_Dashboard.pdf)  

It summarizes:  
- **Overall NPS & CSAT**  
- **Monthly Trends**  
- **Top 10 Products by NPS**  
- **Bottom 10 Products by NPS**  

---

## 🚀 How to Use
1. Clone the repo:
   ```bash
   git clone https://github.com/rgao77/Amazon-FineFood-VoC-Project.git
2. Open the notebook in Jupyter Lab to reproduce the full analysis:
   Amazon Fine Food VoC–NPS Project.ipynb
3. Use the tidy CSV files in Tableau for visualization:
   VoC_metrics_long.csv → KPIs, trends, product metrics
   VoC_topics_with_themes.csv → Themes (pain points & delighters)
