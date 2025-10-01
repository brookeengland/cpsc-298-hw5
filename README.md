# Wikipedia, Bias, and Democracy
### Group Members
- Brooke England
- Carolina Caraballo Velez
## Abstract
Wikipedia has transformed how knowledge is shared and consumed, however, it also raises important questions about
neutrality and bias. This research seeks to investigate how the collaborative nature and openness of Wikipedia reduces
or increases bias compared to traditional encyclopedias. Building upon prior research, some key areas that will be explored 
are the various sources of bias and how we can identify bias at the sentence level and build models to detect biased language.

Our research also explores how Wikipedia reflects democratic practices. Research shows that while it promotes collaboration, small groups of editors often dominate governance (Konieczny, 2009; Shaw & Hill, 2014). Wikipedia’s Talk pages function as deliberative spaces, and politically diverse contributors can improve content quality (Klemp & Forcehimes, 2010; Shi et al., 2019). This project analyzes Wikipedia data to assess neutrality in discussions, examine consensus-building, study governance of contentious topics, and evaluate participation patterns. Wikipedia thus offers insight into both the possibilities and limitations of democracy in online collaborative knowledge production.

## [Literature Review](literature-review.md)

See local file for detailed discussion of research articles, including citations with links to Google Scholar and PDFs

## Research Questions
1. How can we detect and measure biased language in Wikipedia articles at the sentence level?
2. How does Wikipedia bias change over time through the revision process?
3. To what extent do Wikipedia Talk pages function as democratic spaces of deliberation?
4. How do patterns of participation influence consensus-building?
5. Does democratic participation reduce bias, or do small groups of dominant editors shape outcomes that increase bias?

## Methodology
**1. Bias Detection**
- Build upon Hube (2017) and related studies by applying bias lexicon and machine learning models (e.g. Random Forest) to identify biased statements at the sentence level
- Use article revision history to trace how bias changes over time, analyzing who adds or deletes content, what sources are cited, and how phrasing evolves

**2. Democracy & Governance Analysis**
- Analyze talk pages of active controversial articles using LLM-based text analysis to evaluate neutrality and deliberation

**3. Integrating Bias and Democracy**
- Compare article-level bias with Talk page governance structures to investigate connections between inclusiveness in discussions and neutrality in content
- Assess whether diverse, democratic participation reduces bias, or whether small, dominant groups shape articles that enable bias.

## Wikipedia API Prototype

The program fetches the content of a Wikipedia talk page (e.g., “Talk:Climate Change”) using the Wikipedia API, extracts individual editor comments, and analyzes them with two machine learning models: a sentiment analysis model (classifying comments as Positive, Neutral, or Negative) and a toxicity detection model. It handles long comments by chunking, classifies each comment with a confidence level (Low, Medium, High), and aggregates the results to provide distributions and median confidence scores. The program can also visualize these results, giving an overview of the tone, neutrality, and potential bias in the conversation.

_The chart above showcases the analysis on the United States Talk Page_

<img width="1354" height="479" alt="Screenshot 2025-09-30 at 9 41 51 AM" src="https://github.com/user-attachments/assets/b4325c14-71e1-48f7-94a0-0a47d2714060" />

```python
import requests
import re
from transformers import pipeline
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

# ============================
# 1. Fetch talk page content
# ============================
endpoint = "https://en.wikipedia.org/w/api.php"

params = {
    "action": "query",
    "format": "json",
    "prop": "revisions",
    "titles": "Talk:United States",
    "rvprop": "content",
    "rvslots": "main"
}

headers = {"User-Agent": "CarolinaBot/1.0 (https://example.com)"}

response = requests.get(endpoint, params=params, headers=headers)
data = response.json()

pages = data["query"]["pages"]
page = list(pages.values())[0]

if "revisions" not in page:
    print(f"Page '{page['title']}' does not exist or has no revisions.")
    exit()

content = page["revisions"][0]["slots"]["main"]["*"]

# ============================
# 2. Extract comments
# ============================
comment_pattern = re.compile(r"(.*?)(--.*?\d{2}:\d{2},.*?\(UTC\))", re.DOTALL)
matches = comment_pattern.findall(content)
comments = [m[0].strip() for m in matches if m[0].strip()]

print(f"Extracted {len(comments)} comments.\n")

# ============================
# 3. Load ML models
# ============================
sentiment_model = pipeline("sentiment-analysis", model="tabularisai/multilingual-sentiment-analysis")
toxicity_model = pipeline("text-classification", model="unitary/toxic-bert")

# ============================
# 4. Helper functions
# ============================
def classify_long_text(pipeline_model, text, max_length=512):
    words = text.split()
    results = []

    for i in range(0, len(words), max_length):
        chunk = " ".join(words[i:i+max_length])
        result = pipeline_model(chunk, truncation=True, max_length=max_length)[0]
        results.append(result)

    # Choose chunk with highest score
    final = max(results, key=lambda r: r["score"])
    return final

def categorize_score(score):
    """Convert numeric score to Low / Medium / High"""
    if score < 0.5:
        return "Low"
    elif score < 0.75:
        return "Medium"
    else:
        return "High"

# ============================
# 5. Run predictions & collect results
# ============================
sentiment_labels = []
sentiment_confidences = []
toxicity_confidences = []

for i, comment in enumerate(comments, 1):
    sentiment = classify_long_text(sentiment_model, comment)
    toxicity = classify_long_text(toxicity_model, comment)

    sentiment_label = sentiment['label']  # Positive / Neutral / Negative
    sentiment_conf = categorize_score(sentiment['score'])
    toxicity_conf = categorize_score(toxicity['score'])

    sentiment_labels.append(sentiment_label)
    sentiment_confidences.append(sentiment_conf)
    toxicity_confidences.append(toxicity_conf)

    print(f"Comment {i}: {comment[:120]}...")
    print(f"  Sentiment → {sentiment_label} ({sentiment_conf})")
    print(f"  Toxicity  → {toxicity['label']} ({toxicity_conf})")
    print("-" * 60)

# ============================
# 6. Aggregation
# ============================
# Sentiment distribution by label
sentiment_label_dist = Counter(sentiment_labels)
# Sentiment distribution by confidence
sentiment_conf_dist = Counter(sentiment_confidences)
# Toxicity distribution by confidence
toxicity_conf_dist = Counter(toxicity_confidences)

# Median confidence
score_to_num = {"Low": 1, "Medium": 2, "High": 3}
num_to_score = {1: "Low", 2: "Medium", 3: "High"}

median_sentiment_conf = num_to_score[int(np.median([score_to_num[c] for c in sentiment_confidences]))]
median_toxicity_conf = num_to_score[int(np.median([score_to_num[c] for c in toxicity_confidences]))]

print("\n=== Aggregated Results ===")
print(f"Median Sentiment Confidence → {median_sentiment_conf}")
print(f"Median Toxicity Confidence  → {median_toxicity_conf}\n")

print("Sentiment Label Distribution:")
for k in ["Positive", "Neutral", "Negative"]:
    print(f"  {k}: {sentiment_label_dist.get(k,0)}")

print("\nSentiment Confidence Distribution:")
for k in ["Low", "Medium", "High"]:
    print(f"  {k}: {sentiment_conf_dist.get(k,0)}")

print("\nToxicity Confidence Distribution:")
for k in ["Low", "Medium", "High"]:
    print(f"  {k}: {toxicity_conf_dist.get(k,0)}")

# ============================
# 7. Visualization
# ============================
fig, axs = plt.subplots(1, 3, figsize=(18,5))

# Sentiment label
axs[0].bar(sentiment_label_dist.keys(), sentiment_label_dist.values(), color='skyblue')
axs[0].set_title("Sentiment Label Distribution")
axs[0].set_ylabel("Number of Comments")

# Sentiment confidence
axs[1].bar(sentiment_conf_dist.keys(), sentiment_conf_dist.values(), color='lightgreen')
axs[1].set_title("Sentiment Confidence Distribution")
axs[1].set_ylabel("Number of Comments")

# Toxicity confidence
axs[2].bar(toxicity_conf_dist.keys(), toxicity_conf_dist.values(), color='salmon')
axs[2].set_title("Toxicity Confidence Distribution")
axs[2].set_ylabel("Number of Comments")

plt.show()

```
