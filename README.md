# AI-Powered E-Commerce Sentiment Analyzer

A lightweight sentiment analysis pipeline for e-commerce reviews. The core model is a standard-library multinomial Naive Bayes classifier, so it can train and run without external dependencies. Optional API and scraping dependencies are listed in `requirements.txt`.

## Project Structure

```text
sentiment_analyzer/
  model.py             # Trainable sentiment classifier
  preprocessing.py     # Review text cleaning/tokenization
  metrics.py           # Accuracy, precision, recall, F1
  api_clients.py       # Optional OpenAI-compatible sentiment client
scripts/
  train_model.py       # Train and save a model
  evaluate.py          # Evaluate a saved model
  predict.py           # Predict sentiment for one review
data/
  sample_reviews.csv   # Small starter dataset
models/
  .gitkeep             # Saved models go here
```

## Quick Start

Install dashboard dependencies:

```powershell
pip install -r requirements.txt
```

Launch the Streamlit dashboard:

```powershell
streamlit run app.py
```

Train a model:

```powershell
python scripts/train_model.py --data data/sample_reviews.csv --model-out models/sentiment_model.json
```

Evaluate it:

```powershell
python scripts/evaluate.py --data data/sample_reviews.csv --model models/sentiment_model.json
```

Predict one review:

```powershell
python scripts/predict.py --model models/sentiment_model.json --text "The delivery was late but the product quality is excellent."
```

## Dataset Format

CSV files should include:

- `review_text`: customer review text
- `sentiment`: one of `positive`, `negative`, or `neutral`

Extra columns such as `rating`, `author`, `product_id`, and `review_date` are allowed.

## Optional API Sentiment

Set these in `.env` or your shell if you want to use an OpenAI-compatible API client:

```text
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=your_api_key
OPENAI_MODEL=gpt-4o-mini
```

The local model is best for a clean portfolio demo. The API client is useful for bootstrapping labels, summarizing feedback, and comparing local predictions against an LLM.
