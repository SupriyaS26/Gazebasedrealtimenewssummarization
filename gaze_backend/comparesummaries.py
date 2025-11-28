import json
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer

# ========= Load JSON =========
with open("output.json", "r", encoding="utf-8") as f:
    data = json.load(f)

articles = [d["article"] for d in data]
custom_summaries = [d["summary_custom"] for d in data]
static_summaries = [d["summary_static"] for d in data]

# ========= Initialize Metrics =========
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

rouge1_custom = []
rouge1_static = []
rougeL_custom = []
rougeL_static = []
cosine_custom = []
cosine_static = []

# TF-IDF vectorizer for cosine similarity
vectorizer = TfidfVectorizer()

for i in range(len(data)):
    art = articles[i]
    custom = custom_summaries[i]
    static = static_summaries[i]

    # ----- ROUGE -----
    r_custom = scorer.score(art, custom)
    r_static = scorer.score(art, static)

    rouge1_custom.append(r_custom["rouge1"].fmeasure)
    rouge1_static.append(r_static["rouge1"].fmeasure)

    rougeL_custom.append(r_custom["rougeL"].fmeasure)
    rougeL_static.append(r_static["rougeL"].fmeasure)

    # ----- Cosine Similarity -----
    vectors = vectorizer.fit_transform([art, custom])
    cosine_custom.append(cosine_similarity(vectors[0], vectors[1])[0][0])

    vectors = vectorizer.fit_transform([art, static])
    cosine_static.append(cosine_similarity(vectors[0], vectors[1])[0][0])


# ========= Plot Graphs =========

x = range(1, len(data) + 1)

plt.figure(figsize=(12, 6))
plt.plot(x, rouge1_custom, label="ROUGE-1: Custom vs Article", marker="o")
plt.plot(x, rouge1_static, label="ROUGE-1: Static vs Article", marker="o")
plt.title("ROUGE-1 Comparison")
plt.xlabel("Article Number")
plt.ylabel("ROUGE-1 Score")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(x, rougeL_custom, label="ROUGE-L: Custom vs Article", marker="o")
plt.plot(x, rougeL_static, label="ROUGE-L: Static vs Article", marker="o")
plt.title("ROUGE-L Comparison")
plt.xlabel("Article Number")
plt.ylabel("ROUGE-L Score")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(x, cosine_custom, label="Cosine: Custom vs Article", marker="o")
plt.plot(x, cosine_static, label="Cosine: Static vs Article", marker="o")
plt.title("Cosine Similarity Comparison")
plt.xlabel("Article Number")
plt.ylabel("Cosine Score")
plt.legend()
plt.grid(True)
plt.show()
