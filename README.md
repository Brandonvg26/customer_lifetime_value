# Customer Lifetime Value — Segmentation + Vector Similarity Search


**Stack:** pandas 2.x · XGBoost 2.x · scikit-learn 1.5 · MLflow 2.16
· SHAP 0.45 · pgvector 0.3 · PostgreSQL 16 · Docker


## What this project demonstrates
End-to-end retail ML: RFM feature engineering → CLV regression →
customer segmentation → vector storage in pgvector → live similarity queries.
Focus is on business interpretation at every step, not just model training.


## Architecture
1. RFM Engineering — compute Recency/Frequency/Monetary from raw transactions
2. CLV Prediction — XGBoost regressor, MLflow tracking, SHAP explanation
3. Segmentation + pgvector — KMeans clusters, vectors stored in PostgreSQL
4. Similarity Search — cosine similarity queries for lookalike targeting
5. Business Interpretation — segment labels, revenue share, action recommendations


## Key decisions
- Online Retail I (not II): II was used in a prior project; I is a distinct dataset
- StandardScaler before KMeans: RFM is unscaled by default; Monetary dominates
  distance metrics without scaling
- pgvector over ChromaDB: runs inside PostgreSQL which most companies already
  operate; no separate vector DB infrastructure required
- Cosine similarity over Euclidean: direction (behavioral pattern) matters more
  than magnitude (spend level) for lookalike identification
- R2 not the primary metric: CLV prediction has high inherent randomness;
  directional ranking is the business goal, not exact spend prediction


## How to run
```bash
python -m venv clv-venv && .\clv-venv\Scripts\Activate.ps1
pip install -r requirements.txt
cd docker && docker compose up -d    # PostgreSQL + pgvector
mlflow ui                            # http://localhost:5000
# Run notebooks 01 → 02 → 03 → 04 → 05 in Jupyter
```


## Dataset
UCI Online Retail I — 541,909 transactions, UK retailer, 2010-2011.
License: CC BY 4.0. Downloaded programmatically in Notebook 1.
