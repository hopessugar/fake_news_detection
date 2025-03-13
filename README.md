# Fake News Detection using Sentiment Analysis

## Overview
This project implements a machine learning model to detect fake news by analyzing text content and sentiment features. It combines natural language processing techniques with sentiment analysis to identify potentially misleading news articles.

## Features
- Text preprocessing pipeline (lowercase conversion, special character removal, tokenization, stopword removal, lemmatization)
- Sentiment feature extraction (polarity and subjectivity)
- TF-IDF vectorization for text representation
- Random Forest classification algorithm
- Feature importance analysis
- Model evaluation with classification metrics
- Simple interface for testing new articles

## Requirements
- Python 3.7+
- pandas
- numpy
- scikit-learn
- nltk
- textblob
- matplotlib
- seaborn

## Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/fake-news-detection.git
cd fake-news-detection

# Install required packages
pip install -r requirements.txt

# Download required NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

## Usage
### Training the model
```python
from fake_news_detector import fake_news_detector

# Train the model
model, tfidf = fake_news_detector()
```

### Making predictions
```python
from fake_news_detector import predict_news

# Predict on a new article
title = "Scientists discover new treatment for cancer"
content = "A team of researchers has published findings about a potential new approach to treating specific types of cancer."

result, confidence, sentiment = predict_news(model, tfidf, title, content)
print(f"Prediction: {result} (Confidence: {confidence:.2f})")
```

## Datasets
The model can be trained on various fake news datasets, including:
1. [Kaggle Fake News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
2. [LIAR Dataset](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip)
3. [FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet)

## How It Works
1. **Data Preprocessing**: Cleans and normalizes text data
2. **Feature Extraction**: 
   - Converts text to TF-IDF vectors
   - Extracts sentiment features (polarity, subjectivity)
3. **Training**: Trains a Random Forest classifier on the combined features
4. **Evaluation**: Assesses model performance using classification metrics
5. **Prediction**: Makes predictions on new articles with confidence scores

## Model Evaluation
The model performance can be evaluated using:
- Classification report (precision, recall, F1-score)
- Confusion matrix
- Feature importance analysis

## Improvements and Extensions
- Implement more advanced NLP techniques (word embeddings, transformers)
- Add more linguistic features (readability scores, emotional tone)
- Incorporate source credibility information
- Create a web interface for easy testing
- Implement model explainability tools

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- TextBlob for sentiment analysis
- NLTK for text preprocessing
- scikit-learn for machine learning algorithms
