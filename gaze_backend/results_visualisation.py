# ============================================================
# RESULTS VISUALISATION FOR GAZE-BASED REAL-TIME SUMMARIZATION
# ============================================================

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb

from scipy.stats import skew, kurtosis

# ============================================================
# 1. LOAD DATA
# ============================================================

with open("output.json", "r", encoding="utf-8") as f:
    data = json.load(f)

articles = [d["article"] for d in data]
custom_summaries = [d["summary_custom"] for d in data]
static_summaries = [d["summary_static"] for d in data]

# ============================================================
# 2. METRIC INITIALIZATION
# ============================================================

scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
vectorizer = TfidfVectorizer()

rouge1_custom, rouge1_static = [], []
rougeL_custom, rougeL_static = [], []
cosine_custom, cosine_static = [], []

# ============================================================
# 3. METRIC COMPUTATION
# ============================================================

for art, custom, static in zip(articles, custom_summaries, static_summaries):

    # ---- ROUGE ----
    r_custom = scorer.score(art, custom)
    r_static = scorer.score(art, static)

    rouge1_custom.append(r_custom["rouge1"].fmeasure)
    rougeL_custom.append(r_custom["rougeL"].fmeasure)

    rouge1_static.append(r_static["rouge1"].fmeasure)
    rougeL_static.append(r_static["rougeL"].fmeasure)

    # ---- Cosine Similarity ----
    vecs = vectorizer.fit_transform([art, custom])
    cosine_custom.append(cosine_similarity(vecs[0], vecs[1])[0][0])

    vecs = vectorizer.fit_transform([art, static])
    cosine_static.append(cosine_similarity(vecs[0], vecs[1])[0][0])

x = np.arange(1, len(data) + 1)

# ============================================================
# 4. LINE PLOTS
# ============================================================

plt.figure()
plt.plot(x, rouge1_custom, marker="o", label="Gaze-based T5")
plt.plot(x, rouge1_static, marker="o", label="Static T5")
plt.xlabel("Article Number")
plt.ylabel("ROUGE-1")
plt.title("ROUGE-1 Comparison")
plt.legend()
plt.show()

plt.figure()
plt.plot(x, rougeL_custom, marker="o", label="Gaze-based T5")
plt.plot(x, rougeL_static, marker="o", label="Static T5")
plt.xlabel("Article Number")
plt.ylabel("ROUGE-L")
plt.title("ROUGE-L Comparison")
plt.legend()
plt.show()

plt.figure()
plt.plot(x, cosine_custom, marker="o", label="Gaze-based T5")
plt.plot(x, cosine_static, marker="o", label="Static T5")
plt.xlabel("Article Number")
plt.ylabel("Cosine Similarity")
plt.title("Cosine Similarity Comparison")
plt.legend()
plt.show()

# ============================================================
# 5. BOX & VIOLIN PLOTS
# ============================================================

plt.figure()
plt.boxplot([rougeL_custom, rougeL_static], labels=["Gaze-based", "Static"])
plt.ylabel("ROUGE-L")
plt.title("ROUGE-L Distribution")
plt.show()

plt.figure()
plt.violinplot([cosine_custom, cosine_static], showmeans=True)
plt.xticks([1, 2], ["Gaze-based", "Static"])
plt.ylabel("Cosine Similarity")
plt.title("Cosine Similarity Distribution")
plt.show()

# ============================================================
# 6. HEATMAP (CORRELATION)
# ============================================================

metrics_custom = np.array([rouge1_custom, rougeL_custom, cosine_custom])
metrics_static = np.array([rouge1_static, rougeL_static, cosine_static])

corr_custom = np.corrcoef(metrics_custom)
corr_static = np.corrcoef(metrics_static)

labels = ["ROUGE-1", "ROUGE-L", "Cosine"]

plt.figure()
plt.imshow(corr_custom)
plt.colorbar()
plt.xticks(range(3), labels)
plt.yticks(range(3), labels)
plt.title("Correlation Heatmap (Gaze-based)")
plt.show()

plt.figure()
plt.imshow(corr_static)
plt.colorbar()
plt.xticks(range(3), labels)
plt.yticks(range(3), labels)
plt.title("Correlation Heatmap (Static)")
plt.show()

# ============================================================
# 7. REGRESSION ANALYSIS (NO TRAIN–TEST SPLIT)
# ============================================================

# Feature matrix
X = np.column_stack([
    rouge1_custom,
    rougeL_custom,
    cosine_custom
])

# Target: attention proxy
y = np.array(rougeL_custom)

models = {
    "CB Regression": LinearRegression(),
    "STE Regression": SVR(kernel="rbf"),
    "RF Regression": RandomForestRegressor(n_estimators=50),
    "XGB Regression": xgb.XGBRegressor(objective="reg:squarederror")
}

results = {}
stats = {}

for name, model in models.items():
    model.fit(X, y)
    preds = model.predict(X)

    r2 = r2_score(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    mae = mean_absolute_error(y, preds)

    results[name] = [r2, rmse, mae]

    errors = y - preds
    stats[name] = [
        np.mean(errors),
        np.median(errors),
        np.std(errors),
        skew(errors),
        kurtosis(errors)
    ]

metrics_df = pd.DataFrame(
    results, index=["R2", "RMSE", "MAE"]
).T

stats_df = pd.DataFrame(
    stats,
    index=["Mean", "Median", "Std Dev", "Skewness", "Kurtosis"]
).T

print("\nRegression Performance:")
print(metrics_df)

print("\nError Statistics:")
print(stats_df)

# ============================================================
# 8. RADAR PLOT
# ============================================================

labels = metrics_df.columns.tolist()
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]

plt.figure(figsize=(7, 7))
ax = plt.subplot(111, polar=True)

for model in metrics_df.index:
    values = metrics_df.loc[model].tolist()
    values += values[:1]
    ax.plot(angles, values, label=model)
    ax.fill(angles, values, alpha=0.1)

ax.set_thetagrids(np.degrees(angles[:-1]), labels)
ax.set_title("Regression Model Comparison")
ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.1))
plt.show()
