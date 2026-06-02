# Bangla Fake News Detection

A machine learning and deep learning project for detecting fake news in Bangla text. The project uses a labelled Bangla news dataset containing headlines, article content, categories, and labels. It includes exploratory data analysis, Bangla text preprocessing, TF-IDF feature extraction, multiple classical machine learning models, and a BERT-based classifier.

## Project Overview

Fake news spreads quickly through online platforms and can mislead readers, damage trust, and create social confusion. This project aims to classify Bangla news articles as either **authentic news** or **fake news** using natural language processing and supervised learning.

The notebook explores the dataset, visualises text patterns, cleans Bangla text, trains several machine learning models, evaluates their performance, and tests prediction on custom Bangla news text.

## Dataset

The project uses the dataset:

```text
fakenewsdataset - 3045.csv
```

The dataset contains **3,044 records** and **4 columns**:

| Column | Description |
|---|---|
| `category` | News category, such as politics, sports, national, international, entertainment, technology, etc. |
| `headline` | Bangla news headline |
| `content` | Full Bangla news content/article body |
| `label` | Target class for classification |

### Label Meaning

| Label | Meaning |
|---|---|
| `0` | Fake news |
| `1` | Authentic news |

### Dataset Distribution

| Label | Count |
|---|---:|
| Authentic news `1` | 1,900 |
| Fake news `0` | 1,144 |

## Project Files

```text
.
├── fake_news_detection.ipynb      # Main Jupyter/Google Colab notebook
├── fakenewsdataset - 3045.csv     # Bangla fake news dataset
├── README.md                      # Project documentation
```

After running the notebook, the following model files may also be generated:

```text
tfidf_vectorizer.pkl
logistic_regression_model.pkl
naive_bayes_model.pkl
svm_model.pkl
random_forest_model.pkl
knn_model.pkl
decision_tree_model.pkl
gradient_boosting_model.pkl
adaboost_model.pkl
xgboost_model.pkl
best_bert_model.pt
```

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- WordCloud
- BNLP Toolkit
- Scikit-learn
- XGBoost
- PyTorch
- Hugging Face Transformers
- Joblib
- Jupyter Notebook / Google Colab

## Main Workflow

### 1. Install Required Libraries

The notebook installs required packages such as:

```python
!pip install xgboost
!pip install bnlp-toolkit==1.2.0
!pip install bnlp_toolkit
!pip install wordcloud
```

### 2. Load Dataset

```python
import pandas as pd

df = pd.read_csv('fakenewsdataset - 3045.csv')
print(df.columns)
```

> Note: In the notebook, some cells may use a Google Colab path such as `/content/fakenewsdataset - 1652.csv`. If you are using the attached dataset, update the path to:

```python
df = pd.read_csv('/content/fakenewsdataset - 3045.csv')
```

or, if running locally:

```python
df = pd.read_csv('fakenewsdataset - 3045.csv')
```

### 3. Exploratory Data Analysis

The notebook performs basic analysis such as:

- Separating authentic and fake news
- Checking headline length
- Checking content length
- Plotting bar charts and distribution graphs
- Creating Bangla word clouds
- Finding frequent words in authentic and fake news

### 4. Bangla Text Preprocessing

The project uses the `BasicTokenizer` from the BNLP toolkit for Bangla tokenisation.

Text cleaning includes:

- Tokenisation
- Stopword removal
- Punctuation removal
- Digit removal
- Combining cleaned headline and content

Example preprocessing function:

```python
def clean_text(text):
    tokens = btokenizer.tokenize(text)
    filtered = []

    for token in tokens:
        if token in stopwords:
            continue
        if token in punctuations:
            continue
        if token in digits:
            continue
        filtered.append(token)

    return " ".join(filtered)
```

### 5. Feature Extraction

For classical machine learning models, the project uses TF-IDF vectorisation:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
X_train_tfidf = tfidf.fit_transform(X_train_text)
X_test_tfidf = tfidf.transform(X_test_text)
```

The vectorizer is saved using Joblib:

```python
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
```

### 6. Machine Learning Models

The notebook trains and evaluates the following models:

- Logistic Regression
- Naive Bayes
- Support Vector Machine
- Random Forest
- K-Nearest Neighbours
- Decision Tree
- Gradient Boosting
- AdaBoost
- XGBoost

### 7. Model Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Classification report
- Confusion matrix
- ROC curve

### 8. BERT-Based Classifier

The project also includes a deep learning model using multilingual BERT:

```python
bert-base-multilingual-cased
```

The BERT classifier uses:

- Hugging Face `AutoTokenizer`
- Hugging Face `AutoModel`
- PyTorch `Dataset` and `DataLoader`
- Dropout layer
- Linear classification layer

## Model Performance

The following results were produced in the notebook using TF-IDF features and classical machine learning models:

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.8367 | 0.7927 | 0.9934 | 0.8818 |
| Naive Bayes | 0.8266 | 0.7795 | 1.0000 | 0.8761 |
| SVM | 0.8185 | 0.8110 | 0.9178 | 0.8611 |
| Random Forest | 0.8286 | 0.7829 | 0.9967 | 0.8770 |
| KNN | 0.8226 | 0.8121 | 0.9243 | 0.8646 |
| Decision Tree | 0.8145 | 0.8419 | 0.8586 | 0.8502 |
| Gradient Boosting | 0.8266 | 0.8114 | 0.9342 | 0.8685 |
| AdaBoost | 0.8105 | 0.8125 | 0.8980 | 0.8531 |
| XGBoost | 0.8226 | 0.7983 | 0.9507 | 0.8679 |

The BERT model was trained for 3 epochs and achieved a validation accuracy of approximately **82.48%**.

## Custom Prediction Example

The notebook includes a custom fake news prediction example:

```text
Headline: ঘোড়া এখন কথা বলতে পারে
Content: ঘোড়া এখন কথা বলতে পারে — এই খবরটি স্থানীয় সূত্রে জানা গেলেও এর কোনো বৈজ্ঞানিক ভিত্তি নেই।
Prediction: Fake News
```

## How to Run the Project

### Option 1: Run in Google Colab

1. Open `fake_news_detection.ipynb` in Google Colab.
2. Upload `fakenewsdataset - 3045.csv` to the Colab environment.
3. Update the dataset path if needed.
4. Run the notebook cells from top to bottom.

### Option 2: Run Locally

1. Clone or download the project folder.
2. Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn wordcloud scikit-learn xgboost joblib torch transformers bnlp-toolkit
```

3. Open the notebook:

```bash
jupyter notebook fake_news_detection.ipynb
```

4. Run all cells.

## Important Notes

- The project is focused on Bangla text classification.
- Some file paths in the notebook are written for Google Colab. Update them based on your local or Colab environment.
- The notebook uses Bangla font support for word cloud and visualisation.
- The dataset is imbalanced, with more authentic news than fake news.
- Model performance can be improved by better preprocessing, more balanced data, hyperparameter tuning, and using Bangla-specific transformer models.

## Possible Future Improvements

- Use a larger and more balanced Bangla fake news dataset.
- Apply more advanced Bangla text preprocessing.
- Compare multilingual BERT with Bangla-specific models such as BanglaBERT.
- Add hyperparameter tuning for each machine learning model.
- Build a simple web application for real-time fake news detection.
- Save the best-performing model and deploy it using Flask, FastAPI, or Streamlit.
- Add explainable AI methods to understand why a news article is classified as fake or authentic.

## Project Outcome

This project demonstrates how Bangla fake news detection can be performed using both classical machine learning and transformer-based deep learning techniques. The best classical models achieved around **81%–84% accuracy**, while the BERT-based approach achieved around **82% validation accuracy** in the notebook experiment.

## Author

Prepared for a Bangla Fake News Detection research/project workflow.
