# IMDB Movie Reviews: Sentiment Classification Pipeline 🎬🍿

### 📌 Project Overview
This project builds an automated Natural Language Processing (NLP) pipeline to classify the sentiment polarity (positive/negative) of movie reviews from the IMDB database. It evaluates multiple text representation strategies—ranging from classic frequency-based statistical engineering (TF-IDF) to deep sequence transformers (BERT)—paired with linear classifiers and gradient-boosted decision trees.

### 🛠️ Technical Stack & Architecture
* **Data Processing & Analytics:** Python, Pandas, NumPy, Regex, Matplotlib, Seaborn.
* **NLP & Feature Engineering:** NLTK, Stopwords Optimization, TF-IDF Vectorization, HuggingFace Transformers (BERT-base-uncased Tokenizer).
* **Machine Learning & Deep Learning Frameworks:** PyTorch, Scikit-learn, LightGBM (LGBMClassifier), Multinomial Naive Bayes, Logistic Regression.

### 📊 Model Architecture Matrix & Iterations
1. **Model 0 (Baseline):** Stratified Dummy Classifier to establish foundational threshold.
2. **Model 1 (Linear Benchmark):** TF-IDF Vectorizer (NLTK stopwords filtered) + Logistic Regression.
3. **Model 2 (Bayesian Flow):** TF-IDF Vectorizer + Multinomial Naive Bayes.
4. **Model 3 (Optimized Linear Pipeline):** Fast tokenization regex pipeline + high-iteration Logistic Regression.
5. **Model 4 (Ensemble Method):** High-dimensional TF-IDF sparse matrices processed via LightGBM Decision Trees.
6. **Model 9 (Deep Learning Pipeline):** Feature extraction via pre-trained **BERT Transformer** layers by capturing the localized text semantics of the `[CLS]` token, mapped into a Logistic Regression linear head.

### 🏆 Operational Safeguards & Optimizations
* **Memory Leakage Protection:** Integrated a programmatic barrier (mini-batch chunking) during BERT feature generation to guarantee execution without exceeding cloud system memory limits.
* **API Attribute Patching:** Refactored the transformer sequence handling by replacing deprecated indexing with explicit `.last_hidden_state` property calls, securing the codebase against deployment-level crashes.
* **Deterministic Tracking:** Enforced rigid structural alignment across vectorizer mappings to guarantee absolute validation and custom production inference consistency.
