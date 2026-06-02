<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&height=220&color=0:0F2027,50:203A43,100:2C5364&text=Bangla%20Fake%20News%20Detection&fontColor=FFFFFF&fontSize=42&fontAlignY=38&desc=NLP%20%7C%20Machine%20Learning%20%7C%20Multilingual%20BERT&descAlignY=58&descSize=18" />
</p>

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=22&duration=2500&pause=800&color=00D9FF&center=true&vCenter=true&width=900&lines=Detecting+Fake+News+in+Bangla+Text;TF-IDF+%2B+Classical+Machine+Learning;Transformer-Based+Classification+with+Multilingual+BERT;Built+for+Bangla+NLP+Research" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/NLP-Bangla%20Text-00C853?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/BERT-Transformer-FF6F00?style=for-the-badge" />
</p>

<p align="center">
  <b>A complete Bangla fake news classification pipeline using NLP, TF-IDF, classical machine learning models, and multilingual BERT.</b>
</p>

<br>

---

## Project Overview

This project builds a Bangla fake news detection system that classifies Bangla news articles as either fake or authentic.

<table align="center">
  <tr>
    <th>Label</th>
    <th>Class</th>
    <th>Meaning</th>
  </tr>
  <tr>
    <td align="center"><b>0</b></td>
    <td align="center">Fake News</td>
    <td>Misleading or false Bangla news content</td>
  </tr>
  <tr>
    <td align="center"><b>1</b></td>
    <td align="center">Authentic News</td>
    <td>Real and reliable Bangla news content</td>
  </tr>
</table>

<br>

The system combines Bangla text preprocessing, TF-IDF vectorization, multiple classical machine learning classifiers, and a multilingual BERT-based deep learning model to detect fake news from Bangla headlines and article content.

---

## Project Highlights

<table>
  <tr>
    <td><b>Bangla NLP Focused</b></td>
    <td>Designed specifically for Bangla news text classification.</td>
  </tr>
  <tr>
    <td><b>Text Preprocessing</b></td>
    <td>Tokenization, stopword removal, punctuation cleaning, and digit removal.</td>
  </tr>
  <tr>
    <td><b>Exploratory Data Analysis</b></td>
    <td>Headline/content length analysis, class distribution, and word frequency visualization.</td>
  </tr>
  <tr>
    <td><b>TF-IDF Feature Extraction</b></td>
    <td>Unigram and bigram TF-IDF representation for classical machine learning models.</td>
  </tr>
  <tr>
    <td><b>Machine Learning Models</b></td>
    <td>Logistic Regression, SVM, Random Forest, XGBoost, Naive Bayes, and more.</td>
  </tr>
  <tr>
    <td><b>BERT Classifier</b></td>
    <td>Transformer-based text classification using <code>bert-base-multilingual-cased</code>.</td>
  </tr>
</table>

---

## Dataset

The project uses the following dataset:

```text
fakenewsdataset - 3045.csv
```

### Dataset Summary

<table align="center">
  <tr>
    <th>Total Records</th>
    <th>Total Columns</th>
    <th>Language</th>
    <th>Task</th>
  </tr>
  <tr>
    <td align="center"><b>3,044</b></td>
    <td align="center"><b>4</b></td>
    <td align="center"><b>Bangla</b></td>
    <td align="center"><b>Binary Classification</b></td>
  </tr>
</table>

### Dataset Columns

| Column     | Description                                                                                      |
| ---------- | ------------------------------------------------------------------------------------------------ |
| `category` | News category such as politics, sports, national, international, entertainment, technology, etc. |
| `headline` | Bangla news headline                                                                             |
| `content`  | Full Bangla news article body                                                                    |
| `label`    | Target class: fake or authentic                                                                  |

### Class Distribution

| Class          | Label |   Count |
| -------------- | ----: | ------: |
| Authentic News |   `1` | `1,900` |
| Fake News      |   `0` | `1,144` |

---

## System Architecture

```mermaid
flowchart LR
    A[Bangla News Dataset] --> B[Exploratory Data Analysis]
    B --> C[Bangla Text Cleaning]
    C --> D[Headline and Content Merge]

    D --> E[TF-IDF Vectorization]
    E --> F[Classical Machine Learning Models]

    D --> G[BERT Tokenization]
    G --> H[Multilingual BERT]

    F --> I[Model Evaluation]
    H --> I

    I --> J[Fake or Authentic Prediction]
```

---

## Technology Stack

<p align="center">
  <img src="https://skillicons.dev/icons?i=python,pytorch,sklearn,github" />
</p>

| Area                 | Tools                           |
| -------------------- | ------------------------------- |
| Programming Language | Python                          |
| Data Processing      | Pandas, NumPy                   |
| Visualization        | Matplotlib, Seaborn, WordCloud  |
| NLP                  | BNLP Toolkit, TF-IDF            |
| Machine Learning     | Scikit-learn, XGBoost           |
| Deep Learning        | PyTorch                         |
| Transformer Model    | Hugging Face Transformers       |
| Environment          | Jupyter Notebook / Google Colab |
| Model Saving         | Joblib                          |

---

## Models Used

<table>
  <tr>
    <th>Category</th>
    <th>Models</th>
  </tr>
  <tr>
    <td><b>Classical Machine Learning</b></td>
    <td>
      Logistic Regression, Naive Bayes, Support Vector Machine, Random Forest,
      K-Nearest Neighbours, Decision Tree, Gradient Boosting, AdaBoost, XGBoost
    </td>
  </tr>
  <tr>
    <td><b>Deep Learning</b></td>
    <td>
      Multilingual BERT: <code>bert-base-multilingual-cased</code>
    </td>
  </tr>
</table>

---

## Performance Leaderboard

| Rank | Model               | Accuracy | Precision |   Recall | F1-score |
| ---: | ------------------- | -------: | --------: | -------: | -------: |
|    1 | Logistic Regression | `0.8367` |  `0.7927` | `0.9934` | `0.8818` |
|    2 | Random Forest       | `0.8286` |  `0.7829` | `0.9967` | `0.8770` |
|    3 | Naive Bayes         | `0.8266` |  `0.7795` | `1.0000` | `0.8761` |
|    4 | Gradient Boosting   | `0.8266` |  `0.8114` | `0.9342` | `0.8685` |
|    5 | XGBoost             | `0.8226` |  `0.7983` | `0.9507` | `0.8679` |
|    6 | KNN                 | `0.8226` |  `0.8121` | `0.9243` | `0.8646` |
|    7 | SVM                 | `0.8185` |  `0.8110` | `0.9178` | `0.8611` |
|    8 | Decision Tree       | `0.8145` |  `0.8419` | `0.8586` | `0.8502` |
|    9 | AdaBoost            | `0.8105` |  `0.8125` | `0.8980` | `0.8531` |

---

## Best Classical Model

<p align="center">
  <img src="https://img.shields.io/badge/Best%20Model-Logistic%20Regression-00C853?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Accuracy-83.67%25-00D9FF?style=for-the-badge" />
  <img src="https://img.shields.io/badge/F1--Score-88.18%25-FFB300?style=for-the-badge" />
</p>

---

## BERT Result

The multilingual BERT model was trained for 3 epochs and achieved the following validation accuracy:

<p align="center">
  <img src="https://img.shields.io/badge/BERT%20Validation%20Accuracy-82.48%25-FF6F00?style=for-the-badge&logo=pytorch&logoColor=white" />
</p>

---

## Sample Prediction

```text
Headline:
ঘোড়া এখন কথা বলতে পারে

Content:
ঘোড়া এখন কথা বলতে পারে — এই খবরটি স্থানীয় সূত্রে জানা গেলেও এর কোনো বৈজ্ঞানিক ভিত্তি নেই।

Prediction:
Fake News
```

---

## Repository Structure

```text
bangla-fake-news-detection/
│
├── fake_news_detection.ipynb
├── fakenewsdataset - 3045.csv
├── README.md
├── .gitignore
│
├── tfidf_vectorizer.pkl
├── logistic_regression_model.pkl
├── naive_bayes_model.pkl
├── svm_model.pkl
├── random_forest_model.pkl
├── knn_model.pkl
├── decision_tree_model.pkl
├── gradient_boosting_model.pkl
├── adaboost_model.pkl
├── xgboost_model.pkl
└── best_bert_model.pt
```

---


---

## Core Pipeline

```python
# Load dataset
df = pd.read_csv("fakenewsdataset - 3045.csv")

# Combine headline and content
df["clean_text"] = df["headline"] + " " + df["content"]

# TF-IDF feature extraction
tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=10000
)

# Train and transform data
X_train_tfidf = tfidf.fit_transform(X_train_text)
X_test_tfidf = tfidf.transform(X_test_text)
```

---

## Future Enhancements

* Use a larger and more balanced Bangla fake news dataset
* Add Bangla-specific transformer models such as BanglaBERT
* Improve Bangla text normalization
* Apply hyperparameter tuning
* Add explainable AI techniques for prediction interpretation
* Build a Streamlit or Flask web application
* Deploy the model as a real-time Bangla fake news detection system

---

## Author

<p align="center">
  <b>Saiful Islam Priyan</b>
  <br>
  Bangla NLP | Machine Learning | Fake News Detection
</p>

<p align="center">
  <a href="https://github.com/priyansaiful">
    <img src="https://img.shields.io/badge/GitHub-priyansaiful-181717?style=for-the-badge&logo=github" />
  </a>
</p>

---

## Support This Project

<p align="center">
  <b>If this project helps you, consider giving it a star on GitHub.</b>
</p>

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&height=140&section=footer&color=0:2C5364,50:203A43,100:0F2027" />
</p>
