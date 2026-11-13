
import os
import pandas as pd

# Create output folder
os.makedirs("Dataset/processed", exist_ok=True)

# Load datasets
users = pd.read_csv("Dataset/users.csv")
products = pd.read_csv("Dataset/products.csv")
interactions = pd.read_csv(
    "Dataset/interactions.csv",
    parse_dates=["timestamp"]
)

print("STARTING DATA PREPROCESSING...")

# 1. Remove duplicate records
interactions = interactions.drop_duplicates().copy()

# 2. Handle missing values
interactions = interactions.dropna(
    subset=["user_id", "product_id", "event_type", "timestamp"]
)

# 3. Validate user and product IDs
interactions = interactions[
    interactions["user_id"].isin(users["user_id"])
    & interactions["product_id"].isin(products["product_id"])
].copy()

# 4. Normalize event names
interactions["event_type"] = (
    interactions["event_type"].astype(str).str.strip().str.lower()
)

# 5. Assign implicit feedback weights
event_weights = {
    "view": 1,
    "cart": 3,
    "purchase": 5
}

interactions["interaction_weight"] = (
    interactions["event_type"].map(event_weights)
)

# Remove unsupported event types
interactions = interactions.dropna(
    subset=["interaction_weight"]
)

interactions["interaction_weight"] = (
    interactions["interaction_weight"].astype(int)
)

# 6. Sort chronologically
interactions = interactions.sort_values(
    "timestamp"
).reset_index(drop=True)

# 7. Create chronological train-test split
split_position = int(len(interactions) * 0.80)
cutoff_time = interactions.loc[
    split_position, "timestamp"
]

train_data = interactions[
    interactions["timestamp"] < cutoff_time
].copy()

test_data = interactions[
    interactions["timestamp"] >= cutoff_time
].copy()

# 8. Save processed datasets
interactions.to_csv(
    "Dataset/processed/cleaned_interactions.csv",
    index=False
)

train_data.to_csv(
    "Dataset/processed/train_interactions.csv",
    index=False
)

test_data.to_csv(
    "Dataset/processed/test_interactions.csv",
    index=False
)

# 9. Display results
print("\nPREPROCESSING COMPLETED!")
print("----------------------------------")

print("Cleaned interactions:", len(interactions))
print("Training interactions:", len(train_data))
print("Testing interactions:", len(test_data))

print("\nTraining date range:")
print(train_data["timestamp"].min(), "to",
      train_data["timestamp"].max())

print("\nTesting date range:")
print(test_data["timestamp"].min(), "to",
      test_data["timestamp"].max())

print("\nTraining event distribution:")
print(train_data["event_type"].value_counts())

print("\nTesting event distribution:")
print(test_data["event_type"].value_counts())

print("\nTraining users:", train_data["user_id"].nunique())
print("Testing users:", test_data["user_id"].nunique())

print("\nTraining products:", train_data["product_id"].nunique())
print("Testing products:", test_data["product_id"].nunique())

# 10. Verify no temporal leakage
assert train_data["timestamp"].max() < test_data["timestamp"].min()

print("\nLeakage check: PASSED")
print("All training interactions occur before testing interactions.")

print("\nProcessed files saved in Dataset/processed/")
