
import random
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. GENERATE SYNTHETIC SUPPORT TICKET DATASET (600+ rows)
random.seed(42)

templates = [
    # Billing
    ("My invoice #{id} shows double charges for this month", "Billing", "High"),
    ("Need a copy of my tax receipt for order #{id}", "Billing", "Low"),
    ("Credit card payment failed when renewing subscription", "Billing", "High"),
    ("How do I update my billing address on my account?", "Billing", "Low"),
    ("Refund for cancelled plan #{id} has not arrived yet", "Billing", "Medium"),
    ("Charged after cancelling my free trial yesterday", "Billing", "Medium"),
    # Technical Support
    ("Application crashes immediately on login screen", "Technical", "High"),
    ("API returning 500 internal server error on endpoint /users", "Technical", "High"),
    ("Export to CSV button is unresponsive in Safari", "Technical", "Medium"),
    ("Two-factor authentication SMS code is delayed", "Technical", "Medium"),
    ("Dark mode toggle resets after refreshing the page", "Technical", "Low"),
    ("Dashboard charts are loading slowly today", "Technical", "Low"),
    # Account Management
    ("Locked out of my account after resetting password", "Account", "High"),
    ("Want to transfer workspace ownership to another admin", "Account", "Medium"),
    ("Please delete my personal data and close account #{id}", "Account", "Medium"),
    ("How do I change my registered email address?", "Account", "Low"),
    ("Add three new team seats to our company profile", "Account", "Low"),
    ("Suspicious login alert received from unknown location", "Account", "High"),
    # Feature Request
    ("Please add Slack integration for real-time alerts", "Feature Request", "Low"),
    ("Would love a mobile app version for iOS and Android", "Feature Request", "Low"),
    ("Need bulk user import via Excel spreadsheet", "Feature Request", "Medium"),
    ("Support custom webhooks for ticket status updates", "Feature Request", "Medium"),
]

prefixes = ["Hi team,", "Hello,", "Urgent:", "Support,", ""]
suffixes = ["Please help.", "Thanks!", "Need this resolved soon.", "ASAP", ""]

rows = []
for i in range(900):
    text, category, priority = random.choice(templates)
    ticket_text = f"{random.choice(prefixes)} {text.format(id=random.randint(1000, 9999))} {random.choice(suffixes)}"
    rows.append({"text": ticket_text, "category": category, "priority": priority})

df = pd.DataFrame(rows)
raw_count = len(df)

# 2. CLEAN THE DATA (lowercase, trim whitespace, remove duplicates)
df["text"] = df["text"].str.lower().str.strip()
df["text"] = df["text"].str.replace(r"\s+", " ", regex=True)
df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

# 3. CREATE COMPOSITE STRATIFICATION KEY (stratify by BOTH category & priority)
df["strat_key"] = df["category"] + "_" + df["priority"]

# 4. STRATIFIED SPLIT: 70% Train / 15% Validation / 15% Test
train_df, temp_df = train_test_split(
    df, test_size=0.30, random_state=42, stratify=df["strat_key"]
)

val_df, test_df = train_test_split(
    temp_df, test_size=0.50, random_state=42, stratify=temp_df["strat_key"]
)

for split_df in (train_df, val_df, test_df):
    split_df.drop(columns=["strat_key"], inplace=True)

# 5. SAVE PROCESSED DATASETS
df.drop(columns=["strat_key"]).to_csv("data/processed/tickets_clean.csv", index=False)
train_df.to_csv("data/processed/train.csv", index=False)
val_df.to_csv("data/processed/val.csv", index=False)
test_df.to_csv("data/processed/test.csv", index=False)

# 6. PRINT DATASET SUMMARY FOR SUBMISSION
print("=== DATASET SUMMARY ===")
print(f"Raw rows generated : {raw_count}")
print(f"Clean unique rows  : {len(df)}")
print(f"Train split (70%)  : {len(train_df)}")
print(f"Val split (15%)    : {len(val_df)}")
print(f"Test split (15%)   : {len(test_df)}\n")

print("--- Category Class Balance ---")
print(df["category"].value_counts(), "\n")

print("--- Priority Class Balance ---")
print(df["priority"].value_counts())
