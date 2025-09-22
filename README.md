# Amazon Fine Food – Voice of Customer (VoC) Analytics Project

## Project Goal
This project analyzes Amazon Fine Food Reviews to uncover customer insights.  
The focus is on **Net Promoter Score (NPS), CSAT, CES, sentiment, and topic modeling**  
to identify **key pain points** (detractors) and **delighters** (promoters).  
The ultimate goal is to demonstrate how Voice of Customer (VoC) data can drive  
**customer-centric business strategy**.

---

##  Methods
- **Data Cleaning & Feature Engineering**  
  - Processed 500K+ reviews (sampled for efficiency).  
  - Built NPS labels (Promoter / Passive / Detractor).  
  - Created features: helpfulness ratio, text length, exclamation count, capitalization ratio, sentiment score, CES keywords.  

- **Metrics Framework**  
  - Calculated NPS & CSAT overall, by month, and by product.  
  - Applied winsorization & minimum thresholds for robust metrics.  

- **Modeling**  
  - Logistic Regression to distinguish Promoters vs. Detractors.  
  - Identified top positive and negative drivers (keywords + numeric features).  

- **Topic Modeling (LDA)**  
  - Extracted recurring themes from reviews.  
  - Pain point themes (detractors) vs. delight themes (promoters).  

- **Visualization**  
  - Exported tidy CSVs for Tableau dashboards.  
  - Built interactive charts: NPS/CSAT trends, product leaders/laggards, topic clusters.

---

##  Deliverables
- **Jupyter Notebook**: Full end-to-end pipeline (cleaning → metrics → modeling → topics → export).  
- **Tidy Data for Tableau**  
  - `VoC_metrics_long.csv` → KPIs, trends, product-level metrics.  
  - `VoC_topics_with_themes.csv` → Pain points & Delighters (themes + keywords).  
- **Visualization**: Tableau dashboard (example screenshots or Tableau Public link can be added).  

---

##  Dataset
The project uses the **Amazon Fine Food Reviews Dataset** available on Kaggle:  
 [Amazon Fine Food Reviews (Kaggle)](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)

> ️ Note: The raw dataset is large (≈500 MB) and not included in this repo.  
> Instead, this repo contains processed CSVs (tidy format) for easy Tableau use.

---

##  Key Insights (Example)
- **Overall NPS**: +42, **CSAT**: 4.2 / 5  
- **Top Detractor Themes**: packaging issues, stale food, delivery delays.  
- **Top Promoter Themes**: freshness, taste, fast shipping.  
- Logistic Regression highlighted **keywords like “fresh”, “delicious” (positive)** vs. **“stale”, “broken”, “refund” (negative)**.  

---

##  How to Use
1. Clone this repo:
   ```bash
   git clone https://github.com/rgao77/Amazon-FineFood-VoC-Project.git

