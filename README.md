# Support Ticket Classifier

## Dataset Description
- **Source:** Programmatically generated synthetic customer support dataset (`src/prepare_data.py`) simulating realistic multi-department helpdesk tickets.
- **Size:** 536 unique tickets (minimum 500 required) with three columns:
  - `text`: Cleaned customer support inquiry (lowercased, whitespace-trimmed, deduplicated).
  - `category`: Department routing label (`Billing`, `Technical`, `Account`, `Feature Request`).
  - `priority`: Urgency level (`Low`, `Medium`, `High`).
- **Preprocessing & Split:** Text was normalized to lowercase, extra spaces were trimmed, and exact duplicate texts were dropped. The dataset was split into **70% Train / 15% Validation / 15% Test** using composite stratification (`category` + `priority`) to preserve class distributions across all splits.
## Baseline Model
- **Pipeline:** `TfidfVectorizer` (`sublinear_tf=True`) + `LogisticRegression`, tuned via 5-fold stratified cross-validation (`ngram_range` and `C`).
- **Validation Results:**
  - **Category:** Accuracy `1.00`, Macro F1 `1.00` (`ngram_range=(1, 1)`, `C=1.0`)
  - **Priority:** Accuracy `1.00`, Macro F1 `1.00` (`ngram_range=(1, 1)`, `C=0.1`)
- **Error Analysis & Artifact:** Due to clean synthetic templates, validation accuracy reached `1.00`; the 10 lowest-margin validation tickets were analyzed against their runner-up classes. Saved trained pipelines to `models/baseline_tfidf_logreg.joblib`.
## Model Comparison & Deployment Selection

| Model | Accuracy | Macro F1 | Training time | Inference time per ticket |
| :--- | :---: | :---: | :---: | :---: |
| TF-IDF + LogReg (Baseline) | 1.0000 | 1.0000 | 0.10 s | 0.03 ms |
| DistilBERT (Fine-Tuned) | 1.0000 | 1.0000 | <paste_distilbert_time> s | 40.67 ms |

### Deployment Justification
1. Deploy TF-IDF + Logistic Regression for production routing because it matches DistilBERT's accuracy and macro F1 on structured support tickets.
2. DistilBERT wins on unseen paraphrases, negation, and subtle contextual phrasing where bag-of-words n-grams lack semantic representation.
3. The TF-IDF baseline is sufficient here because ticket categories rely on clear domain keywords ('invoice', '2FA', 'crash', 'feature') that linear n-grams separate cleanly.
4. On speed and cost, TF-IDF trains in 0.10s and serves in ~0.03 ms/ticket on standard CPU (~1272x faster than DistilBERT at 40.67 ms/ticket), avoiding GPU hosting costs.
5. Reserve fine-tuned DistilBERT as a Stage-2 fallback only for low-confidence tickets (top-2 margin < 0.20) or when expanding to noisy multi-lingual/free-form customer chat.