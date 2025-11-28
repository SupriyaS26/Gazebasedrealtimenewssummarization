
import json
import re
import spacy
import feedparser
from bs4 import BeautifulSoup
from custom_t5 import CustomT5
import re
import matplotlib.pyplot as plt
from transformers import T5ForConditionalGeneration, T5Tokenizer

#Load SpaCy model
nlp = spacy.load("en_core_web_sm")

#Fetch and parse news articles from RSS feeds
def fetch_rss_news(urls):
    
    articles = []
    for url in urls:
        feed = feedparser.parse(url)
        for entry in feed.entries:
            content = entry.title + ". " + entry.get("summary", "")
            articles.append(content)
    return articles

#Remove HTML tags and extra whitespace
def clean_html(text):
    if not text:
        return ""
    soup = BeautifulSoup(text, "html.parser")
    clean = soup.get_text(separator=" ")
    clean = re.sub(r'\s+', ' ', clean)
    clean = re.sub(r'\s+([.!?])', r'\1', clean)  
    return clean.strip()

#Compute simple saliency scores based on named entities
def extract_saliency_scores(sentences):
    
    scores = []
    for sent in sentences:
        doc = nlp(sent)
        sal = sum([1 for _ in doc.ents])
        scores.append(sal / max(1, len(doc)))
    return scores

#Identify sentences containing key named entities
def extract_key_facts(sentences):
    
    key_sents = []
    for sent in sentences:
        doc = nlp(sent)
        ents = {ent.label_ for ent in doc.ents}
        if ents.intersection({"PERSON", "ORG", "GPE", "DATE", "TIME", "EVENT"}):
            key_sents.append(sent)
    return key_sents

#Load gaze scores for a specific article (by article index)
def load_gaze_scores_for_article(filepath, article_index, num_sentences):
    
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
    except:
        return [0.0] * num_sentences

    
    article_gaze_data = [item for item in data if item.get("line", -1) == article_index]
    
    if not article_gaze_data:
        
        return [0.0] * num_sentences
    
    
    avg_gaze_score = sum([item.get("gaze_score", 0) for item in article_gaze_data]) / len(article_gaze_data)
    
    
    gaze_per_sentence = [avg_gaze_score] * num_sentences
    
    return gaze_per_sentence

def capitalize_sentences(text):
    if not text:
        return ""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s[:1].upper() + s[1:] if s else s for s in sentences]
    return ' '.join(sentences)

def validate_and_clean_summary(summary, source_article):
    
    if not summary or len(summary.strip()) < 10:
        
        sentences = re.split(r'(?<=[.!?])\s+', source_article)
        if len(sentences) >= 2:
            return ". ".join(sentences[:2]) + "."
        return source_article[:200] + "..." if len(source_article) > 200 else source_article
    
    summary_lower = summary.lower()
    source_lower = source_article.lower()
    
    
    source_doc = nlp(source_article[:1000]) 
    source_entities = set()
    for ent in source_doc.ents:
        if ent.label_ in ["GPE", "LOC", "PERSON", "ORG"]:
            source_entities.add(ent.text.lower())
    
    
    hallucination_patterns = [
        r'\bin the (u\.?s\.?|united states|usa)\b',
        r'\bin (egypt|uk|britain|france|germany)\b',
        r'\b(u\.?s\.?|usa|egypt|uk|britain)\s+(incident|event|marriage|stadium|olympic)',
        r'\bafter (his|her|their) death\b',
        r'\bhe said after\b',
        r'\bstring of.*marriages?\b',
    ]
    
   
    summary_sentences = re.split(r'(?<=[.!?])\s+', summary)
    cleaned_sentences = []
    
    for sent in summary_sentences:
        sent = sent.strip()
        if not sent or len(sent) < 5:
            continue
        
        
        has_hallucination = False
        for pattern in hallucination_patterns:
            if re.search(pattern, sent.lower()):
               
                if not re.search(pattern, source_lower):
                    has_hallucination = True
                    break
        
        
        sent_words = set([w.lower().strip('.,!?;:') for w in sent.split() if len(w) > 3])
        source_words = set([w.lower().strip('.,!?;:') for w in source_article.split() if len(w) > 3])
        overlap = len(sent_words.intersection(source_words))
        
       
        if not has_hallucination and (overlap >= 2 or len(sent.split()) < 6):
            cleaned_sentences.append(sent)
    
    if cleaned_sentences:
        result = ". ".join(cleaned_sentences)
        if result and result[-1] not in ".!?":
            result += "."
        return result
    else:
        
        sentences = re.split(r'(?<=[.!?])\s+', source_article)
        if len(sentences) >= 2:
            return ". ".join(sentences[:3]) + "."
        return source_article[:200] + "..." if len(source_article) > 200 else source_article

#Generate gaze-adaptive summary for a single article
def summarize_article(article, t5_model, gaze_scores):
    
    article_clean = clean_html(article)
    if not article_clean or len(article_clean.strip()) < 50:
        return article_clean, 0.0
    
    
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', article_clean)
    
    
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    
    
    if len(sentences) < 2:
        sentences = [s.strip() + '.' for s in article_clean.split('.') if s.strip() and len(s.strip()) > 10]

    if not sentences or len(sentences) == 0:
        
        sentences = [article_clean]

    
    if len(gaze_scores) != len(sentences):
        
        if len(gaze_scores) < len(sentences):
            gaze_scores = gaze_scores + [0.0] * (len(sentences) - len(gaze_scores))
        else:
            gaze_scores = gaze_scores[:len(sentences)]

    saliency_scores = extract_saliency_scores(sentences)
    attention_scores = t5_model.apply_attention_layers(sentences, gaze_scores, saliency_scores)

    
    avg_attention = round(sum(attention_scores) / max(1, len(attention_scores)), 2)

    
    summary = t5_model.summarize(sentences, attention_scores, max_length=120)

   
    if not summary or len(summary) < 30:
        
        sorted_idx = sorted(range(len(sentences)), key=lambda i: attention_scores[i], reverse=True)
        top_sentences = [sentences[i] for i in sorted_idx[:min(3, len(sentences))]]
        summary = ". ".join(top_sentences)
        if summary and summary[-1] not in ".!?":
            summary += "."

    return summary, avg_attention


def filter_long_articles(articles, min_words=50):
    
    filtered = []
    for article in articles:
        article_clean = clean_html(article)
        
        word_count = len(article_clean.split())
        
        if word_count >= min_words:
            filtered.append(article)
        else:
            print(f"Filtered out short article: {article_clean[:50]}... ({word_count} words)")
    return filtered

def main():
    try:
        with open("news_raw.json", "r", encoding="utf-8") as f:
            raw_articles = json.load(f)
           
            articles = []
            for a in raw_articles:
                
                article_text = a.get("description", "")
                if not article_text and a.get("title"):
                   
                    article_text = a["title"] + ". " + a.get("description", "")
                if article_text:
                    articles.append(article_text)
            
            
            articles = filter_long_articles(articles, min_words=50)
            print(f"After filtering: {len(articles)} articles with sufficient length (>=50 words)")
            
    except Exception as e:
        print("Using live RSS fetch because news_raw.json not found:", e)
        rss_urls = [
            "https://timesofindia.indiatimes.com/rssfeeds/-2128936835.cms",
            "https://feeds.feedburner.com/ndtvnews-top-stories"
        ]
        articles = fetch_rss_news(rss_urls)
       
        articles = filter_long_articles(articles, min_words=50)
    t5_model = CustomT5(model_name="t5-small")
    output_data = []
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    static_t5 = T5ForConditionalGeneration.from_pretrained("t5-small")


   
    try:
        with open("gaze_scores.json", "r") as f:
            all_gaze_data = json.load(f)
        
        article_indices_with_gaze = set([item.get("line", -1) for item in all_gaze_data if item.get("line", -1) >= 0])
        print(f"Found gaze data for articles at indices: {sorted(article_indices_with_gaze)}")
    except:
        article_indices_with_gaze = set(range(len(articles)))  
        print("Could not load gaze_scores.json, will process all articles")
    
    processed_count = 0
    
    for article_index, article in enumerate(articles):
        if processed_count >= 10:  
            break
        
        
        if article_indices_with_gaze and article_index not in article_indices_with_gaze:
            continue
            
        article_clean = clean_html(article)
        
        if len(article_clean.split()) < 30:
            print(f"Skipping article {article_index} with only {len(article_clean.split())} words")
            continue
        
        sentences = re.split(r'(?<=[.!?]) +', article_clean)
        if len(sentences) < 3:
            print(f"Skipping article {article_index} with only {len(sentences)} sentences")
            continue
            
       
        gaze_scores = load_gaze_scores_for_article("gaze_scores.json", article_index, len(sentences))
        summary, attention_score = summarize_article(article, t5_model, gaze_scores)
        
        
        summary = validate_and_clean_summary(summary, article_clean)
        summary = capitalize_sentences(summary)
        
        inputs = tokenizer.encode("summarize: " + article_clean, 
                          return_tensors="pt", truncation=True, max_length=512)

       
        generate_kwargs = {
            "max_length": 120,
            "min_length": 40,
            "num_beams": 8,
            "early_stopping": True,
            "length_penalty": 1.5,
            "no_repeat_ngram_size": 2,
            "do_sample": False,
        }
        
        
        import inspect
        sig = inspect.signature(static_t5.generate)
        if 'repetition_penalty' in sig.parameters:
            generate_kwargs["repetition_penalty"] = 1.2
        
        outputs = static_t5.generate(inputs, **generate_kwargs)
        static_summary = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        
       
        static_summary = validate_and_clean_summary(static_summary, article_clean)
        
        
        if static_summary and static_summary[-1] not in ".!?":
            static_summary += "."
        if static_summary and len(static_summary) > 0:
            static_summary = static_summary[0].upper() + static_summary[1:] if len(static_summary) > 1 else static_summary.upper()

        output_data.append({
            "article": article_clean,
            "summary_custom": summary,
            "summary_static": static_summary,
            "attention_score": attention_score
        })
        
        processed_count += 1

    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print("Summaries saved to output.json")
    #methods = ['Static T5', 'Gaze-T5']
    #from rouge_score import rouge_scorer

    #scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    #score_static = scorer.score(original_text, static_summary)['rougeL'].fmeasure
    #score_gaze = scorer.score(original_text, gaze_summary)['rougeL'].fmeasure

    #rouge_scores = [score_static, score_gaze]
    #rouge_scores = [0.58, 0.72]
    #compression = [0.35, 0.30]

    #x = range(len(methods))
    #plt.bar(x, rouge_scores, width=0.4, label='ROUGE-L', align='center')
    #plt.bar([i + 0.4 for i in x], compression, width=0.4, label='Compression Ratio')
    #plt.xticks([i + 0.2 for i in x], methods)
    #plt.ylabel("Scores")
    #plt.title("Static vs Gaze-Based Summarization Comparison")
    #plt.legend()
    #plt.show()

if __name__ == "__main__":
    main()
