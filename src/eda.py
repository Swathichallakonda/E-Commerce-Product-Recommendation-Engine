
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create folder for EDA charts
os.makedirs("screenshots", exist_ok=True)

# Load datasets
users = pd.read_csv("Dataset/users.csv")
products = pd.read_csv("Dataset/products.csv")
interactions = pd.read_csv(
    "Dataset/interactions.csv",
    parse_dates=["timestamp"]
)

print("\n========== DATASET OVERVIEW ==========")

print("\nUsers shape:", users.shape)
print("Products shape:", products.shape)
print("Interactions shape:", interactions.shape)

print("\n========== DATA QUALITY CHECK ==========")

print("\nMissing values:")
print("Users:\n", users.isnull().sum())
print("\nProducts:\n", products.isnull().sum())
print("\nInteractions:\n", interactions.isnull().sum())

print("\nDuplicate interaction records:",
      interactions.duplicated().sum())

print("\nUnique users:",
      interactions["user_id"].nunique())

print("Unique products interacted with:",
      interactions["product_id"].nunique())

print("\n========== INTERACTION ANALYSIS ==========")

event_counts = interactions["event_type"].value_counts()
print(event_counts)

print("\n========== PRODUCT ANALYSIS ==========")

print("\nProducts by category:")
print(products["category"].value_counts())

print("\nProduct price statistics:")
print(products["price"].describe())

print("\n========== USER ANALYSIS ==========")

print("\nUsers by preferred category:")
print(users["preferred_category"].value_counts())

print("\n========== INTERACTIONS OVER TIME ==========")

monthly_interactions = (
    interactions.set_index("timestamp")
    .resample("ME")
    .size()
)

print(monthly_interactions)

# -----------------------------
# Chart 1: Interaction types
# -----------------------------
plt.figure(figsize=(8, 5))
event_counts.plot(kind="bar")
plt.title("Distribution of Interaction Types")
plt.xlabel("Interaction Type")
plt.ylabel("Number of Interactions")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("screenshots/interaction_distribution.png")
plt.close()

# -----------------------------
# Chart 2: Product categories
# -----------------------------
plt.figure(figsize=(9, 5))
products["category"].value_counts().plot(kind="bar")
plt.title("Products by Category")
plt.xlabel("Category")
plt.ylabel("Number of Products")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("screenshots/product_categories.png")
plt.close()

# -----------------------------
# Chart 3: Product price distribution
# -----------------------------
plt.figure(figsize=(8, 5))
sns.histplot(products["price"], bins=25, kde=True)
plt.title("Product Price Distribution")
plt.xlabel("Price")
plt.ylabel("Number of Products")
plt.tight_layout()
plt.savefig("screenshots/product_price_distribution.png")
plt.close()

# -----------------------------
# Chart 4: Monthly interactions
# -----------------------------
plt.figure(figsize=(9, 5))
monthly_interactions.plot(kind="line", marker="o")
plt.title("Monthly Interaction Trend")
plt.xlabel("Month")
plt.ylabel("Number of Interactions")
plt.tight_layout()
plt.savefig("screenshots/monthly_interactions.png")
plt.close()

print("\nEDA COMPLETED SUCCESSFULLY!")
print("Four charts saved in the screenshots folder.")
