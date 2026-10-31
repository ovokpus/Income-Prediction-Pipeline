# Business case: Income Prediction Pipeline

**A complete train, track, serve and monitor loop for a tabular scoring model.**

## A note before the case

The model in this repo predicts whether a person earns more than $50,000 a year, using the 1994 US Census extract known as the UCI Adult dataset (48,842 records; [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/2/adult)). **That model should not make decisions about real people.** The data is decades old, it includes sex and race, and using predicted income to decide on credit, housing or employment raises fair-lending and discrimination issues. The business value here is the **MLOps pipeline around the model**, which carries over to any tabular scoring problem. This case is written on that basis.

## Who buys this

The buyer is the head of data science or the ML platform lead at a company that already has a few tabular models (churn, propensity, risk or demand scores) running from notebooks or ad hoc scripts. They need those models to be reproducible, served behind an API, and watched so they notice when a model goes stale.

## The problem and what it costs

Models that work in a notebook decay quietly in production. Input data drifts, nobody can say which training run produced the live model, and retraining is a manual task. Gartner research puts the average cost of poor data quality at **at least US$12.9 million a year per organisation**. The figure comes from a 2020 survey of 154 reference customers of data-quality vendors, so it is a self-reported estimate from a narrow sample ([Gartner](https://www.gartner.com/en/data-analytics/topics/data-quality)). It is quoted here as context, not as the saving this repo delivers.

## How the repo solves it

| Need | What in the repo does it |
|---|---|
| Reproducible training and model selection | XGBoost and scikit-learn experiments tracked in MLflow (`B_experimentTracking/`) |
| Scheduled retraining | Prefect flows (`C_trainingPipelineOrchestration/`) |
| An API other systems can call | A Flask + gunicorn prediction service in Docker (`prediction_service/`). Every score is stored in MongoDB. |
| Knowing when the model goes stale | Evidently drift metrics scraped by Prometheus and shown in Grafana (`evidently_service/`) |
| A load test with realistic traffic | `stream-generator/` |
| A way to try it | A Streamlit form (`frontend/`) |

## Use-case scenarios

All of these organisations are fictional, and in each one the census model is swapped for the company's own.

1. **A subscription retailer's churn score.** The retention team gets a daily churn probability per customer from the prediction API. When a pricing change shifts customer behaviour, the Evidently dashboard shows feature drift within days, and the Prefect flow retrains on fresh data.
2. **A utility's equipment-failure score.** Field data from sensors feeds the same serve-and-monitor path. MLflow records which model version made each prediction, which the asset-management team needs for audits.
3. **A lender's model-risk team**, *after* replacing the census model with an approved, governed model. They use MLflow lineage plus stored predictions as the evidence for periodic model validation.

## ROI: a worked model (illustrative)

These figures are **illustrative assumptions**.

| Input | Value | Basis |
|---|---|---|
| Production tabular models | 5 | Assumption |
| Data-scientist days per model per year spent on manual retraining, redeploying and investigating "why did scores change?" | 15 | Assumption |
| Share of that time removed by tracked, scheduled retraining and drift dashboards | 50% | Assumption |
| Data-scientist daily cost | US$563 | BLS May 2024 median for data scientists is $112,590 a year, about $54 an hour ([BLS](https://www.bls.gov/ooh/math/data-scientists.htm)). Multiplied by 8 hours and an assumed 1.3 for overhead. |

Saving: 5 × 15 × 50% = **37.5 days, about US$21,000 a year**, before hosting costs. The harder-to-price benefit is catching a stale model before it costs money.

## Path to production

1. Replace the census model and features with the company's own, and remove sensitive attributes unless there is a governed reason to keep them.
2. Upgrade Evidently (the repo pins the 0.1 stack) and move from Flask to a maintained serving stack if needed.
3. Put MLflow and Prefect on managed or shared infrastructure, and add authentication to the prediction API.
4. Define drift thresholds that trigger a retrain or an alert, and test the rollback path.

`PRODUCTIONIZING.md` lists the engineering work.

## Risks and limits

- **This is the weakest of the cases on its own terms.** The demo model has no legitimate production use. The case only holds for the pipeline pattern.
- Several libraries are older versions and will need upgrades.
- Drift dashboards only help if someone owns the alerts.

## Sources

- UCI Machine Learning Repository, Adult dataset: https://archive.ics.uci.edu/dataset/2/adult
- Gartner, Data Quality topic page ($12.9M figure): https://www.gartner.com/en/data-analytics/topics/data-quality
- U.S. Bureau of Labor Statistics, Data Scientists: https://www.bls.gov/ooh/math/data-scientists.htm
