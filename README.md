# Fake News Detection Model

A machine learning-based system that identifies the authenticity of news articles using Natural Language Processing (NLP) techniques.

## Overview

This project builds a binary classifier to distinguish between real and fake news using textual data. The model employs TF-IDF vectorization for text representation and evaluates multiple machine learning classifiers for optimal performance.

## Features

- Text preprocessing (stopword removal, punctuation cleaning)
- TF-IDF vectorization of news content
- Multiple classification algorithms:
  - Passive Aggressive Classifier
  - Naive Bayes
  - Logistic Regression
- Performance evaluation with accuracy metrics, confusion matrices, and classification reports
- Visual comparison of model performance

## Technology Stack8

- Python
- Jupyter Notebook
- Scikit-learn
- Pandas
- Matplotlib/Seaborn
- NLTK

## Project Structure

```
fake-news-detection/
├── FakeNewsDetectionModel.ipynb    # Main analysis notebook
├── README.md                       # Project documentation
├── requirements.txt                # Dependencies
└── data/                          # Dataset directory
    └── fake_news.csv              # News dataset
```

## Model Performance

| Algorithm | Accuracy |
|-----------|----------|
| Passive Aggressive Classifier | ~96% |
| Multinomial Naive Bayes | ~92% |
| Logistic Regression | ~95% |

*Results may vary based on dataset splits and random states*

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/fake-news-detection.git
cd fake-news-detection
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download NLTK data:
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

4. Launch Jupyter Notebook:
```bash
jupyter notebook FakeNewsDetectionModel.ipynb
```

## Requirements

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
nltk>=3.6.0
jupyter>=1.0.0
```

## Dataset

The model expects a CSV file with the following structure:
- `text`: News article content
- `label`: Binary classification (0 = Real, 1 = Fake)

Recommended dataset: [Kaggle Fake News Dataset](https://www.kaggle.com/datasets/c/fake-news)

## Usage

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

# Example prediction
sample_news = "Your news article text here..."
prediction = model.predict([sample_news])
result = "Real News" if prediction[0] == 0 else "Fake News"
print(result)
```

## Methodology

1. Load and inspect dataset
2. Preprocess text data
3. Split into training and testing sets
4. Apply TF-IDF vectorization
5. Train multiple classification models
6. Evaluate and compare performance
7. Visualize results

## Evaluation Metrics

- Accuracy Score
- Confusion Matrix
- Precision and Recall
- F1-Score
- Classification Reports

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/enhancement`)
3. Commit changes (`git commit -m 'Add enhancement'`)
4. Push to branch (`git push origin feature/enhancement`)
5. Open a Pull Request

## License

This project is licensed under the MIT License.

## References

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [NLTK Library](https://www.nltk.org/)
- [TF-IDF Vectorization](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)

## Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your Profile](https://linkedin.com/in/yourprofile)