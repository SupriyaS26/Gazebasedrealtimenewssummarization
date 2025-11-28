import json
from bs4 import BeautifulSoup
from custom_t5 import CustomT5, visualize_attention_layers

# --- Load scraped data ---
with open("news_raw.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# --- Pick the latest news item (based on the first entry) ---
article = data[0]  # You can change index for another article

# --- Clean description text (remove HTML tags, links, and image tags) ---
desc_html = article.get("description", "")
soup = BeautifulSoup(desc_html, "html.parser")
clean_text = soup.get_text().strip()

# --- Split text into sentences ---
sentences = [s.strip() for s in clean_text.split(".") if s.strip()]

if not sentences:
    raise ValueError("No valid sentences extracted from description")

print(f" Title: {article['title']}\n")
print("Extracted Sentences:")
for i, s in enumerate(sentences, 1):
    print(f"{i}. {s}")

# --- Initialize CustomT5 ---
model = CustomT5()

# --- Compute layer scores ---
cldl_scores = model.cldl(sentences)
pfw_scores = model.pfw(sentences)
flpl_scores = model.flpl(sentences)

# --- Compute gaze and saliency from CustomT5 itself ---
gaze_scores = model.compute_gaze_scores(sentences)
saliency_scores = model.compute_saliency_scores(sentences)

# --- Visualize all layers ---
visualize_attention_layers(sentences, cldl_scores, pfw_scores, flpl_scores, gaze_scores, saliency_scores)

# --- Combine for final attention and summarize ---
final_attention = model.apply_attention_layers(sentences, gaze_scores, saliency_scores)
summary = model.summarize(sentences, final_attention)

print("\n Generated Summary:\n", summary)
