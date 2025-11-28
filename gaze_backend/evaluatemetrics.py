import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from collections import Counter
import nltk
import numpy as np

# Download required nltk stuff
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')

# Load output.json
with open("output.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Roug scorer
rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

def repetition_rate(text, n=2):
    """Compute n-gram repetition rate for redundancy."""
    words = text.lower().split()
    ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
    counts = Counter(ngrams)
    repeated = sum(1 for k, v in counts.items() if v > 1)
    total = len(ngrams)
    return repeated / total if total > 0 else 0

results = []

for idx, item in enumerate(data):
    article = item["article"]
    summary_custom = item["summary_custom"]
    summary_static = item["summary_static"]

    ## ROUGE Scores
    rouge_c = rouge.score(article, summary_custom)
    rouge_s = rouge.score(article, summary_static)

    ## BLEU
    smooth = SmoothingFunction().method1
    bleu_custom = sentence_bleu([article.split()], summary_custom.split(), smoothing_function=smooth)
    bleu_static = sentence_bleu([article.split()], summary_static.split(), smoothing_function=smooth)

    ## METEOR
    meteor_custom = meteor_score([article], summary_custom)
    meteor_static = meteor_score([article], summary_static)

    ## Compression Ratio
    comp_custom = len(summary_custom.split()) / len(article.split())
    comp_static = len(summary_static.split()) / len(article.split())

    ## Repetition
    rep_custom = repetition_rate(summary_custom)
    rep_static = repetition_rate(summary_static)

    results.append({
        "article_index": idx+1,

        "rouge1_custom": rouge_c["rouge1"].fmeasure,
        "rouge1_static": rouge_s["rouge1"].fmeasure,

        "rouge2_custom": rouge_c["rouge2"].fmeasure,
        "rouge2_static": rouge_s["rouge2"].fmeasure,

        "rougeL_custom": rouge_c["rougeL"].fmeasure,
        "rougeL_static": rouge_s["rougeL"].fmeasure,

        "bleu_custom": bleu_custom,
        "bleu_static": bleu_static,

        "meteor_custom": meteor_custom,
        "meteor_static": meteor_static,

        "compression_custom": comp_custom,
        "compression_static": comp_static,

        "repetition_custom": rep_custom,
        "repetition_static": rep_static,
    })

# BERTScore (semantic similarity)
all_custom = [item["summary_custom"] for item in data]
all_static = [item["summary_static"] for item in data]
all_articles = [item["article"] for item in data]

P_custom, R_custom, F_custom = bert_score(all_custom, all_articles, lang="en", verbose=True)
P_static, R_static, F_static = bert_score(all_static, all_articles, lang="en", verbose=True)

for i in range(len(results)):
    results[i]["bertscore_custom"] = float(F_custom[i])
    results[i]["bertscore_static"] = float(F_static[i])

# Save results
with open("metrics_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("\n=== METRICS COMPUTED SUCCESSFULLY ===")
print("Saved to metrics_results.json\n")

# Pretty printing
for r in results:
    print(f"Article {r['article_index']}:")
    print(f"  ROUGE-1: custom={r['rouge1_custom']:.3f}, static={r['rouge1_static']:.3f}")
    print(f"  ROUGE-2: custom={r['rouge2_custom']:.3f}, static={r['rouge2_static']:.3f}")
    print(f"  ROUGE-L: custom={r['rougeL_custom']:.3f}, static={r['rougeL_static']:.3f}")
    print(f"  BLEU:    custom={r['bleu_custom']:.3f}, static={r['bleu_static']:.3f}")
    print(f"  METEOR:  custom={r['meteor_custom']:.3f}, static={r['meteor_static']:.3f}")
    print(f"  BERTScore: custom={r['bertscore_custom']:.3f}, static={r['bertscore_static']:.3f}")
    print(f"  Compression: custom={r['compression_custom']:.3f}, static={r['compression_static']:.3f}")
    print(f"  Repetition: custom={r['repetition_custom']:.3f}, static={r['repetition_static']:.3f}")
    print()
