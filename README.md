# Sentiment Analysis of Movie Reviews: ML Project

Classifies IMDB reviews as positive or negative. Compares feature representations and five models
(classical ML + neural network), with evaluation, error analysis and a Streamlit demo.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python src/train.py              # trains all models, writes results/ and models/best_model.joblib
python src/train.py --sample 10000 --cv 0   # quick run while developing
python src/train.py --skip-rf    # skip the slowest model (Random Forest)
streamlit run src/app.py         # web demo
python src/train_lstm.py --sample 20000     # optional LSTM (needs tensorflow)
```

## Pipeline

Raw review -> clean (lowercase, strip HTML, expand "n't", remove stop words but KEEP negations)
-> features (BoW / TF-IDF / bigrams) -> model -> metrics -> best model saved -> Streamlit app.

## Outputs (`results/`)

| File                         | Purpose                                                                  |
| ---------------------------- | ------------------------------------------------------------------------ |
| `feature_comparison.csv`     | BoW vs TF-IDF vs TF-IDF + bigrams                                        |
| `model_comparison.csv/.png`  | Accuracy, precision, recall, F1, CV F1, training time                    |
| `confusion_matrices.png`     | One matrix per model                                                     |
| `top_words.csv`              | Words that push predictions most positive/negative (Logistic Regression) |
| `misclassified_examples.csv` | Material for the error-analysis section of the report                    |
