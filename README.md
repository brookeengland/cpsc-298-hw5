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

# Research Question Week 7
**Question:** How does the overall tone (sentiment and toxicity) of a Wikipedia Talk page vary across different topics?

### Methodology

1. Fetch Talk-page text 
3. Extract editor comments
4. Analyze the tone of each comment
5. Aggregate per Talk page
   - Percentages of positive/neutral/negative comments
   - Percentages of low/medium/high toxicity levels
   - Median confidence scores for both sentiment and toxicity

7. Compare across topics
8. Interpret results

### Wikipedia API Prototype (UPDATED!)
The program still contains the same functionalities from last week, now with the ability to compare several topics, rather than observing the data for just one. Now, the program will ask the user for the number of topics they'd like to compare, and request that each title be input. The program will output 4 graphs: sentiment per comment (positive, neutral, or negative), sentiment confidence (model certainty), toxicity per comment (low, medium, or high), and a stacked bar graph, showing what proportion of comments are positive, neutral, and negative.

_The example below showcases a comparison between the talk pages for Elon Musk, the United States & Civil Rights Movement._

<img width="1510" height="596" alt="Screenshot 2025-10-14 at 11 25 55 PM" src="https://github.com/user-attachments/assets/c2fba47f-c4e6-4ff3-a003-0f90b4808d2e" />

```
import requests
import re
from transformers import pipeline
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

# ===================================
# 1. User Input: Multiple Topics
# ===================================
topics = []
num_topics = int(input("Enter number of Wikipedia talk pages to compare: "))
for i in range(num_topics):
    topic = input(f"Enter topic {i+1} (e.g., United States, Climate change, Elon Musk): ").strip()
    topics.append(topic)

# ===================================
# 2. Wikipedia API Setup
# ===================================
endpoint = "https://en.wikipedia.org/w/api.php"
headers = {"User-Agent": "CarolinaBot/1.0 (https://example.com)"}

# ===================================
# 3. Load Models
# ===================================
print("\nLoading models (this may take a bit)...")
sentiment_model = pipeline("sentiment-analysis", model="tabularisai/multilingual-sentiment-analysis")
toxicity_model = pipeline("text-classification", model="unitary/toxic-bert")

# ===================================
# 4. Helper Functions
# ===================================
def fetch_talk_page_content(topic):
    params = {
        "action": "query",
        "format": "json",
        "prop": "revisions",
        "titles": f"Talk:{topic}",
        "rvprop": "content",
        "rvslots": "main"
    }
    response = requests.get(endpoint, params=params, headers=headers)
    data = response.json()
    page = list(data["query"]["pages"].values())[0]
    if "revisions" not in page:
        print(f"⚠️ No content found for Talk:{topic}")
        return None
    return page["revisions"][0]["slots"]["main"]["*"]

def extract_comments(content):
    comment_pattern = re.compile(r"(.*?)(--.*?\d{2}:\d{2},.*?\(UTC\))", re.DOTALL)
    matches = comment_pattern.findall(content)
    comments = [m[0].strip() for m in matches if m[0].strip()]
    return comments

def classify_long_text(pipeline_model, text, max_length=512):
    words = text.split()
    results = []
    for i in range(0, len(words), max_length):
        chunk = " ".join(words[i:i+max_length])
        result = pipeline_model(chunk, truncation=True, max_length=max_length)[0]
        results.append(result)
    return max(results, key=lambda r: r["score"])

def categorize_score(score):
    """Convert a score (0-1) to Low/Medium/High."""
    if score < 0.5:
        return "Low"
    elif score < 0.75:
        return "Medium"
    else:
        return "High"

# ===================================
# 5. Run Analysis per Topic
# ===================================
topic_results = {}

for topic in topics:
    print(f"\n=== Processing Talk:{topic} ===")
    content = fetch_talk_page_content(topic)
    if not content:
        continue

    comments = extract_comments(content)
    print(f"Extracted {len(comments)} comments.\n")

    sentiment_labels, sentiment_confidences, toxicity_conf_levels = [], [], []

    for comment in comments:
        sentiment = classify_long_text(sentiment_model, comment)
        toxicity = classify_long_text(toxicity_model, comment)

        sentiment_labels.append(sentiment["label"])
        sentiment_confidences.append(categorize_score(sentiment["score"]))
        toxicity_conf_levels.append(categorize_score(toxicity["score"]))  # now as Low/Medium/High

    # Aggregation
    topic_results[topic] = {
        "sentiment_label_dist": Counter(sentiment_labels),
        "sentiment_conf_dist": Counter(sentiment_confidences),
        "toxicity_conf_dist": Counter(toxicity_conf_levels)
    }

# ===================================
# 6. Comparative Visualization
# ===================================
if not topic_results:
    print("No valid topics to display.")
    exit()

topics_list = list(topic_results.keys())
num_topics = len(topics_list)
width = 0.2
x = np.arange(num_topics)

fig, axs = plt.subplots(1, 4, figsize=(28, 6))  # 4 charts side by side

# --- 1. Sentiment Labels (Grouped) ---
labels = ["Positive", "Neutral", "Negative"]
colors = ["#7FC97F", "#BEAED4", "#FDC086"]
for i, label in enumerate(labels):
    counts = [topic_results[t]["sentiment_label_dist"].get(label, 0) for t in topics_list]
    axs[0].bar(x + (i - 1)*width, counts, width, label=label, color=colors[i])
axs[0].set_title("Sentiment Labels")
axs[0].set_xticks(x)
axs[0].set_xticklabels(topics_list, rotation=15)
axs[0].set_ylabel("Number of Comments")
axs[0].legend()

# --- 2. Sentiment Confidence (Grouped) ---
levels = ["Low", "Medium", "High"]
colors_levels = ["#D9D9D9", "#A6CEE3", "#1F78B4"]
for i, level in enumerate(levels):
    counts = [topic_results[t]["sentiment_conf_dist"].get(level, 0) for t in topics_list]
    axs[1].bar(x + (i - 1)*width, counts, width, label=level, color=colors_levels[i])
axs[1].set_title("Sentiment Confidence")
axs[1].set_xticks(x)
axs[1].set_xticklabels(topics_list, rotation=15)
axs[1].legend()

# --- 3. Toxicity Confidence (Grouped) ---
for i, level in enumerate(levels):
    counts = [topic_results[t]["toxicity_conf_dist"].get(level, 0) for t in topics_list]
    axs[2].bar(x + (i - 1)*width, counts, width, label=level, color=colors_levels[i])
axs[2].set_title("Toxicity Levels (Low/Med/High)")
axs[2].set_xticks(x)
axs[2].set_xticklabels(topics_list, rotation=15)
axs[2].legend()

# --- 4. Sentiment Labels (Stacked Percentage) ---
bottom = np.zeros(num_topics)
for i, label in enumerate(labels):
    counts = np.array([topic_results[t]["sentiment_label_dist"].get(label, 0) for t in topics_list])
    topic_totals = np.array([sum(topic_results[t]["sentiment_label_dist"].values()) for t in topics_list])
    perc = np.divide(counts, topic_totals, out=np.zeros_like(counts, dtype=float), where=topic_totals!=0) * 100
    axs[3].bar(x, perc, bottom=bottom, label=label, color=colors[i])
    bottom += perc
axs[3].set_title("Sentiment Labels (Stacked %)")
axs[3].set_xticks(x)
axs[3].set_xticklabels(topics_list, rotation=15)
axs[3].set_ylabel("Percentage (%)")
axs[3].legend()

plt.tight_layout()
plt.show()
```



