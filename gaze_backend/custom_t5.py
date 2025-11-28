# custom_t5.py
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import matplotlib.pyplot as plt
import re
class CustomT5:
    def __init__(self, model_name="t5-small"):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

    def cldl(self, sentences):
        """Custom Layer: CLDL (Content Length Distribution Layer)
        Give higher attention to longer sentences with more information
        """
        scores = [len(s.split()) for s in sentences]
        max_len = max(scores) if scores else 1
        return [s / max_len for s in scores]

    def pfw(self, sentences):
        """Custom Layer: PFW (Position Favors Words)
        Earlier sentences are usually more important
        """
        n = len(sentences)
        return [(n - i)/n for i in range(n)]

    def flpl(self, sentences):
        """Custom Layer: FLPL (Frequency Layer + Punctuation Layer)
        Sentences with more named entities and punctuation get higher scores
        """
        import spacy
        nlp = spacy.load("en_core_web_sm")
        scores = []
        for sent in sentences:
            doc = nlp(sent)
            entity_score = sum([1 for _ in doc.ents])
            punct_score = sum([1 for t in doc if t.is_punct])
            scores.append(entity_score + punct_score)
        max_score = max(scores) if scores else 1
        return [s/max_score for s in scores]

    def apply_attention_layers(self, sentences, gaze_scores, saliency_scores):
        """Combine gaze, saliency, and custom layers to produce attention scores"""
        cldl_scores = self.cldl(sentences)
        pfw_scores = self.pfw(sentences)
        flpl_scores = self.flpl(sentences)
        # Normalize gaze and saliency
        max_gaze = max(gaze_scores) if gaze_scores else 1
        max_sal = max(saliency_scores) if saliency_scores else 1
        #visualize_attention_layers(sentences, cldl_scores, pfw_scores, flpl_scores, gaze_scores, saliency_scores)
        combined = []
        for i in range(len(sentences)):
            gaze_norm = gaze_scores[i]/max_gaze if max_gaze > 0 else 0
            sal_norm = saliency_scores[i]/max_sal if max_sal > 0 else 0
            # Combine all layers with equal weight
            combined.append((gaze_norm + sal_norm + cldl_scores[i] + pfw_scores[i] + flpl_scores[i]) / 5)
        return combined

    def summarize(self, sentences, attention_scores, max_length=100):
        """Generate abstractive summary using top sentences weighted by attention with improved quality"""
        if not sentences:
            return ""

        
        valid_sentences = [(i, s.strip()) for i, s in enumerate(sentences) if len(s.strip()) > 10]
        if not valid_sentences:
            
            valid_sentences = [(i, s.strip()) for i, s in enumerate(sentences) if s.strip()]
        
        if not valid_sentences:
            return ""
        
        valid_indices, valid_sents = zip(*valid_sentences)
        valid_attention = [attention_scores[i] for i in valid_indices]
        
       
        num_sentences = len(valid_sents)
        num_top = min(max(6, int(num_sentences * 0.6)), 12, num_sentences)
        
        
        sorted_pairs = sorted(zip(valid_indices, valid_sents, valid_attention), 
                             key=lambda x: x[2], reverse=True)
        top_pairs = sorted_pairs[:num_top]
        
        
        top_pairs_sorted = sorted(top_pairs, key=lambda x: x[0])
        top_sentences = [s for _, s, _ in top_pairs_sorted]
        
        
        if len(top_sentences) < 3:
            
            top_sentences = valid_sents[:min(5, len(valid_sents))]
        
        
        input_text = " ".join(top_sentences)
        
        
        inputs = self.tokenizer.encode("summarize: " + input_text, 
                                      return_tensors="pt", 
                                      truncation=True, 
                                      max_length=512)
        
        
        generate_kwargs = {
            "max_length": max_length,
            "min_length": 50,  
            "length_penalty": 1.8,  
            "num_beams": 10,  
            "early_stopping": True,
            "no_repeat_ngram_size": 3,  
            "do_sample": False,  
        }
        
       
        import inspect
        sig = inspect.signature(self.model.generate)
        if 'repetition_penalty' in sig.parameters:
            generate_kwargs["repetition_penalty"] = 1.2
        
        outputs = self.model.generate(inputs, **generate_kwargs)
        
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        
       
        if summary:
           
            summary = self._clean_summary(summary, top_sentences)
            
           
            if summary and summary[-1] not in ".!?":
                summary += "."
            
            
            if len(summary) > 0:
                summary = summary[0].upper() + summary[1:] if len(summary) > 1 else summary.upper()
        
        return summary
    
    def _clean_summary(self, summary, source_sentences):
        """Clean summary to reduce hallucinations by checking against source"""
        if not summary:
            return ""
        
       
        source_text = " ".join(source_sentences[:10]) 
        
        
        import spacy
        nlp = spacy.load("en_core_web_sm")
        source_doc = nlp(source_text[:1000]) 
        source_entities = set()
        for ent in source_doc.ents:
            if ent.label_ in ["GPE", "LOC", "PERSON", "ORG", "DATE"]:
                source_entities.add(ent.text.lower())
        
        
        source_words = set()
        for sent in source_sentences[:5]:
            words = sent.lower().split()
            source_words.update([w.strip('.,!?;:') for w in words if len(w) > 3])
        
        summary_sentences = re.split(r'(?<=[.!?])\s+', summary)
        cleaned_sentences = []
        
        
        hallucination_patterns = [
            r'\bin the (u\.?s\.?|united states|usa)\b',
            r'\bin (egypt|uk|britain)\b',
            r'\b(u\.?s\.?|usa|egypt|uk)\s+(incident|event|marriage|stadium)',
            r'\bafter (his|her|their) death\b',
            r'\bhe said after\b',
        ]
        
        for sent in summary_sentences:
            sent = sent.strip()
            if not sent or len(sent) < 5:
                continue
            
            
            has_hallucination = False
            for pattern in hallucination_patterns:
                if re.search(pattern, sent.lower()):
                    
                    sent_doc = nlp(sent)
                    sent_entities = set([ent.text.lower() for ent in sent_doc.ents if ent.label_ in ["GPE", "LOC"]])
                    if not sent_entities.intersection(source_entities):
                        has_hallucination = True
                        break
            
            if has_hallucination:
                continue
            
           
            sent_words = set([w.strip('.,!?;:').lower() for w in sent.split() if len(w) > 3])
            overlap = len(sent_words.intersection(source_words))
            
           
            if overlap > 0 or len(sent.split()) < 5:
                cleaned_sentences.append(sent)
        
        if cleaned_sentences:
            result = '. '.join(cleaned_sentences)
            if result and result[-1] not in '.!?':
                result += '.'
            return result
        else:
           
            return summary
    def compute_gaze_scores(self, sentences):
        
        gaze_scores = np.random.rand(len(sentences))
        gaze_scores = gaze_scores / gaze_scores.sum()
        return gaze_scores.tolist()

    def compute_saliency_scores(self, sentences):
        
        lengths = np.array([len(s.split()) for s in sentences])
        if lengths.sum() == 0:
            return [0] * len(sentences)
        saliency_scores = lengths / lengths.sum()
        return saliency_scores.tolist()

#import numpy as np

#def visualize_attention_layers(sentences, cldl, pfw, flpl, gaze, saliency):
    #layers = ["CLDL", "PFW", "FLPL", "Gaze", "Saliency"]
    #indices = np.arange(len(sentences))
    #width = 0.15  # Bar width

    #plt.figure(figsize=(10, 6))
    #plt.bar(indices - 2*width, cldl, width, label='CLDL')
    #plt.bar(indices - width, pfw, width, label='PFW')
    #plt.bar(indices, flpl, width, label='FLPL')
    #plt.bar(indices + width, gaze, width, label='Gaze')
    #plt.bar(indices + 2*width, saliency, width, label='Saliency')

    #plt.xlabel("Sentence Index")
    #plt.ylabel("Normalized Attention Contribution")
    #plt.title("Attention Layer Contribution per Sentence")
    #plt.legend()
    #plt.tight_layout()
    #plt.show()
    
