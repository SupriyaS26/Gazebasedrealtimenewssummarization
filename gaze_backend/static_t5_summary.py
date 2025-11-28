import json
import matplotlib.pyplot as plt
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import evaluate  # Modern replacement for datasets.load_metric

# Load ROUGE metric
rouge = evaluate.load("rouge")

# Load T5 model
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Load news data
with open("news_raw.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Handle flexible keys (title + description)
news_articles = []
for item in data:
    if "title" in item and "description" in item:
        news_articles.append(item["title"] + ". " + item["description"])
    elif "article" in item:
        news_articles.append(item["article"])
    elif "content" in item:
        news_articles.append(item["content"])

if not news_articles:
    raise ValueError("No valid articles found in news_raw.json. Check field names!")

# Generate static T5 summaries
static_summaries = []
for article in news_articles[:5]:
    input_text = "summarize: " + article
    inputs = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(inputs, max_length=100, min_length=30, length_penalty=1.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    static_summaries.append(summary)

# Save summaries
with open("static_output.json", "w", encoding="utf-8") as f:
    json.dump(static_summaries, f, ensure_ascii=False, indent=2)

print("Static T5 summaries saved to static_output.json")

# Simulate gaze-adaptive summaries from your system
with open("output.json", "r", encoding="utf-8") as f:
    gaze_data = json.load(f)
gaze_summaries = [item["summary"] for item in gaze_data if "summary" in item]

# --- Evaluation ---
if len(static_summaries) == len(gaze_summaries):
    rouge_scores_static = []
    rouge_scores_gaze = []
    compression_static = []
    compression_gaze = []

    for i in range(len(static_summaries)):
        static_rouge = rouge.compute(predictions=[static_summaries[i]], references=[news_articles[i]])["rougeL"]
        gaze_rouge = rouge.compute(predictions=[gaze_summaries[i]], references=[news_articles[i]])["rougeL"]

        rouge_scores_static.append(static_rouge)
        rouge_scores_gaze.append(gaze_rouge)

        compression_static.append(len(static_summaries[i]) / len(news_articles[i]))
        compression_gaze.append(len(gaze_summaries[i]) / len(news_articles[i]))

    # --- Visualization ---
    x = range(1, len(static_summaries) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(x, rouge_scores_static, label="Static T5 ROUGE-L", marker='o')
    plt.plot(x, rouge_scores_gaze, label="Gaze-Adaptive T5 ROUGE-L", marker='s')
    plt.title("ROUGE-L Comparison: Static vs Gaze-Adaptive Summaries")
    plt.xlabel("Article Index")
    plt.ylabel("ROUGE-L Score")
    plt.legend()
    plt.grid(True)
    plt.show()

    print("\nAverage Static ROUGE-L:", round(sum(rouge_scores_static) / len(rouge_scores_static), 3))
    print(" Average Gaze-Adaptive ROUGE-L:", round(sum(rouge_scores_gaze) / len(rouge_scores_gaze), 3))
else:
    print(" Mismatch: static summaries =", len(static_summaries), ", gaze summaries =", len(gaze_summaries))
