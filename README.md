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

  
