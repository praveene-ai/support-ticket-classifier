# Support Ticket Classifier

## Dataset Description
- **Source:** Programmatically generated synthetic customer support dataset (`src/prepare_data.py`) simulating realistic multi-department helpdesk tickets.
- **Size:** 536 unique tickets (minimum 500 required) with three columns:
  - `text`: Cleaned customer support inquiry (lowercased, whitespace-trimmed, deduplicated).
  - `category`: Department routing label (`Billing`, `Technical`, `Account`, `Feature Request`).
  - `priority`: Urgency level (`Low`, `Medium`, `High`).
- **Preprocessing & Split:** Text was normalized to lowercase, extra spaces were trimmed, and exact duplicate texts were dropped. The dataset was split into **70% Train / 15% Validation / 15% Test** using composite stratification (`category` + `priority`) to preserve class distributions across all splits.