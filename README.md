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