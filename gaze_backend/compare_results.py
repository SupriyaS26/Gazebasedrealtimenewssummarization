from static_t5_summary import generate_static_summary
from custom_t5 import generate_dynamic_summary  # your gaze-based summarizer
from rouge_score import rouge_scorer

with open("news_raw.json", "r", encoding="utf-8") as f:
    import json
    data = json.load(f)
    text = data[0]["content"]

# Generate both summaries
static_summary = generate_static_summary(text)
dynamic_summary = generate_dynamic_summary(text, attention_scores=[0.7]*len(text.split('.')))

# ROUGE Comparison
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
scores = scorer.score(static_summary, dynamic_summary)
print("\nROUGE Comparison:", scores)
