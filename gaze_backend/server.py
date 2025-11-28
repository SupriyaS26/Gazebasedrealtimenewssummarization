from flask import Flask, request, jsonify
import xgboost as xgb, pandas as pd, joblib, json, subprocess, os
from flask_cors import CORS
import feedparser
import os, json
from flask import Flask, jsonify
app = Flask(__name__)
CORS(app)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(BASE_DIR, "output.json")

model_x, model_y = None, None

# Gaze Correction 
@app.route('/upload', methods=['POST'])
def upload():
    
    global model_x, model_y
    data = request.json
    df = pd.DataFrame(data)
    X = df[['predX', 'predY', 'scrollY']]
    y_x = df['screenX']
    y_y = df['screenY']

    model_x = xgb.XGBRegressor().fit(X, y_x)
    model_y = xgb.XGBRegressor().fit(X, y_y)
    joblib.dump(model_x, 'x_model.pkl')
    joblib.dump(model_y, 'y_model.pkl')

    return jsonify({'message': 'Model trained', 'samples': len(df)})

@app.route('/predict', methods=['POST'])
def predict():
    
    global model_x, model_y
    if model_x is None or model_y is None:
        if os.path.exists('x_model.pkl') and os.path.exists('y_model.pkl'):
            model_x = joblib.load('x_model.pkl')
            model_y = joblib.load('y_model.pkl')
        else:
            return jsonify({'error': 'Models not trained yet'}), 400

    data = request.json
    df = pd.DataFrame([data])
    corrected_x = model_x.predict(df)[0]
    corrected_y = model_y.predict(df)[0]
    return jsonify({'x': float(corrected_x), 'y': float(corrected_y)})

# Gaze Score Saving 
@app.route('/gaze_scores', methods=['POST'])
def gaze_scores():
    scores = request.get_json(force=True)
    existing = []
    if os.path.exists("gaze_scores.json"):
        with open("gaze_scores.json", "r", encoding="utf-8") as f:
            try:
                existing = json.load(f)
            except:
                existing = []
    existing.extend(scores)
    with open("gaze_scores.json", "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
    return jsonify({"status": "received", "count": len(scores)})

# Run Pipeline 
@app.route('/run_pipeline', methods=['GET'])
def run_pipeline():
    
    try:
        result = subprocess.run(["python", "pipeline.py"], capture_output=True, text=True, check=True)
        return jsonify({"status": "done", "output": result.stdout})
    except subprocess.CalledProcessError as e:
        return jsonify({"status": "error", "message": e.stderr}), 500

# Get Summarized Output 
@app.route('/summarized', methods=['GET'])
def summarized():
    
    if os.path.exists("output.json"):
        try:
            with open("output.json", "r", encoding="utf-8") as f:
                data = json.load(f)
                
                return jsonify({
                    "summarized": data
                })
        except Exception as e:
            print(f"Error loading output.json: {e}")
            return jsonify({
                "summarized": []
            })
    
    return jsonify({
        "summarized": []
    })

# Fetch RSS News 
@app.route('/rss_news', methods=['GET'])
def rss_news():
    urls = [
        "https://timesofindia.indiatimes.com/rssfeeds/-2128936835.cms",
        "https://feeds.feedburner.com/ndtvnews-top-stories"
    ]
    items = []
    try:
        for u in urls:
            try:
                feed = feedparser.parse(u)
                if feed.bozo and feed.bozo_exception:
                    print(f"Warning: RSS feed parsing error for {u}: {feed.bozo_exception}")
                
                
                if not hasattr(feed, 'entries') or len(feed.entries) == 0:
                    print(f"Warning: No entries found in feed {u}")
                    continue
                
                for e in feed.entries[:15]:  
                    
                    desc = e.get("summary", "") or e.get("description", "") or e.get("content", [{}])[0].get("value", "")
                    
                    
                    if e.get("title"):
                        full_text = e.title + ". " + desc
                        word_count = len(full_text.split())
                        if word_count >= 50:  
                            items.append({
                                "title": e.title,
                                "link": e.get("link", ""),
                                "pubDate": e.get("published", "") or e.get("updated", "") or "",
                                "description": desc
                            })
                        
            except Exception as e:
                print(f"Error parsing feed {u}: {str(e)}")
                continue
        
        if len(items) == 0:
            print("Warning: No items collected from any RSS feed")
        
        return jsonify({"items": items})
    except Exception as e:
        print(f"Error in rss_news endpoint: {str(e)}")
        return jsonify({"items": [], "error": str(e)}), 500
@app.route("/save_articles", methods=["POST"])
def save_articles():
    try:
        articles = request.get_json()
        with open("news_raw.json", "w", encoding="utf-8") as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
        return jsonify({"status": "saved", "count": len(articles)})
    except Exception as e:
        return jsonify({"error": str(e)})
# Run Flask 
if __name__ == "__main__":
    app.run(debug=True)
