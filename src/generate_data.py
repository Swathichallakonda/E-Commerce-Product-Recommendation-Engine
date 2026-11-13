
import os
import random
import numpy as np
import pandas as pd

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "Dataset")

os.makedirs(DATASET_DIR, exist_ok=True)

# --------------------------------------------------
# PRODUCT CATALOGUE: EXACTLY 36 PRODUCTS
# Prices are illustrative demo prices in INR.
# --------------------------------------------------

products = [
    # FOOTWEAR (5)
    ("Campus", "Campus Men Running Shoes", "Footwear", "Running Shoes", 899),
    ("PUMA", "Puma Unisex Sports Sneakers", "Footwear", "Sneakers", 2499),
    ("Bata", "Bata Men's Casual Shoes", "Footwear", "Casual Shoes", 1299),
    ("ASIAN", "Asian Women's Walking Shoes", "Footwear", "Walking Shoes", 799),
    ("Campus", "Campus Women's Daily Wear Shoes", "Footwear", "Casual Shoes", 699),

    # MEN'S CLOTHING (5)
    ("H&M", "Men's Regular Fit Cotton T-Shirt", "Men's Clothing", "T-Shirts", 499),
    ("Levi's", "Men's Slim Fit Jeans", "Men's Clothing", "Jeans", 1799),
    ("Roadster", "Men's Casual Checked Shirt", "Men's Clothing", "Shirts", 899),
    ("U.S. Polo Assn.", "Men's Printed Polo T-Shirt", "Men's Clothing", "T-Shirts", 1099),
    ("HRX", "Men's Active Sports Track Pants", "Men's Clothing", "Sportswear", 999),

    # WOMEN'S CLOTHING (5)
    ("Max", "Women's Casual Graphic T-Shirt", "Women's Clothing", "T-Shirts", 399),
    ("Biba", "Women's Printed Kurta", "Women's Clothing", "Kurtas", 899),
    ("ONLY", "Women's Casual Top", "Women's Clothing", "Tops", 799),
    ("Libas", "Women's Printed Ethnic Kurta", "Women's Clothing", "Ethnic Wear", 999),
    ("H&M", "Women's High Waist Jeans", "Women's Clothing", "Jeans", 1499),

    # ACCESSORIES (4)
    ("Fastrack", "Fastrack Analog Wrist Watch", "Accessories", "Watches", 1299),
    ("Wildcraft", "Wildcraft Casual Backpack", "Accessories", "Bags", 1199),
    ("Tommy Hilfiger", "Classic Casual Wallet", "Accessories", "Wallets", 999),
    ("Fastrack", "Fastrack Sunglasses", "Accessories", "Sunglasses", 799),

    # ELECTRONICS (5)
    ("boAt", "boAt Airdopes Wireless Earbuds", "Electronics", "Earbuds", 1499),
    ("JBL", "JBL Wired Headphones", "Electronics", "Headphones", 1299),
    ("Noise", "Noise Smart Fitness Watch", "Electronics", "Smart Watches", 1999),
    ("realme", "realme Wireless Neckband", "Electronics", "Neckbands", 999),
    ("Mivi", "Mivi Portable Bluetooth Speaker", "Electronics", "Speakers", 899),

    # BEAUTY (4)
    ("Maybelline", "Maybelline Liquid Lipstick", "Beauty", "Lipstick", 599),
    ("Lakme", "Lakme Face Concealer", "Beauty", "Makeup", 299),
    ("Plum", "Plum Green Tea Face Wash", "Beauty", "Skincare", 349),
    ("Maybelline", "Maybelline Kajal Eyeliner", "Beauty", "Eye Makeup", 249),

    # HOME & KITCHEN (4)
    ("Milton", "Milton Stainless Steel Water Bottle", "Home & Kitchen", "Bottles", 499),
    ("Cello", "Cello Insulated Lunch Box", "Home & Kitchen", "Lunch Boxes", 599),
    ("Philips", "Philips LED Table Lamp", "Home & Kitchen", "Lighting", 999),
    ("Spaces", "Spaces Cotton Bedsheet Set", "Home & Kitchen", "Bedding", 1299),

    # SPORTS (4)
    ("Nivia", "Nivia Football", "Sports", "Football", 599),
    ("Yonex", "Yonex Badminton Racket", "Sports", "Badminton", 1499),
    ("Nivia", "Nivia Sports Training Gloves", "Sports", "Sports Accessories", 399),
    ("Yonex", "Yonex Shuttlecock Pack", "Sports", "Badminton", 299),
]

# Verify catalogue size before generating anything.
assert len(products) == 36, (
    f"Expected 36 products, but found {len(products)}"
)

# Create product IDs and DataFrame.
products_df = pd.DataFrame(
    [
        {
            "product_id": f"PRD{i + 1:03d}",
            "product_name": item[1],
            "brand": item[0],
            "category": item[2],
            "subcategory": item[3],
            "price": item[4],
            "currency": "INR",
            "image_url": "",
            "product_url": "",
        }
        for i, item in enumerate(products)
    ]
)

# --------------------------------------------------
# USERS: 500 NAMED DEMO USERS
# --------------------------------------------------

first_names = [
    "Aarav", "Aanya", "Aditya", "Ananya", "Arjun",
    "Diya", "Ishaan", "Isha", "Kabir", "Kavya",
    "Krishna", "Meera", "Neha", "Rohan", "Saanvi",
    "Rahul", "Priya", "Sneha", "Varun", "Swara",
    "Vikram", "Anika", "Dev", "Pooja", "Karan"
]

last_names = [
    "Sharma", "Verma", "Patel", "Reddy", "Gupta",
    "Singh", "Kumar", "Mehta", "Rao", "Nair",
    "Chopra", "Malhotra", "Das", "Shah", "Joshi",
    "Iyer", "Kapoor", "Mishra", "Agarwal", "Pillai"
]

categories = [
    "Footwear",
    "Men's Clothing",
    "Women's Clothing",
    "Accessories",
    "Electronics",
    "Beauty",
    "Home & Kitchen",
    "Sports",
]

users_data = []

for i in range(500):
    first = random.choice(first_names)
    last = random.choice(last_names)

    users_data.append({
        "user_id": f"U{i + 1:03d}",
        "user_name": f"{first} {last}",
        "preferred_category": random.choice(categories),
        "age_group": random.choice(
            ["18-25", "26-35", "36-45", "46-55"]
        ),
    })

users_df = pd.DataFrame(users_data)

# --------------------------------------------------
# INTERACTIONS: 30 PER USER = 15,000
# --------------------------------------------------

event_types = ["view", "cart", "purchase"]
event_probabilities = [0.55, 0.22, 0.23]

interaction_rows = []

start_date = pd.Timestamp("2026-01-01")
end_date = pd.Timestamp("2026-06-30")

all_product_ids = products_df["product_id"].tolist()

for user_id in users_df["user_id"]:

    # Select 30 distinct products for each user.
    selected_products = random.sample(all_product_ids, 30)

    for product_id in selected_products:

        event_type = random.choices(
            event_types,
            weights=event_probabilities,
            k=1
        )[0]

        timestamp = start_date + pd.to_timedelta(
            random.randint(
                0,
                int((end_date - start_date).total_seconds())
            ),
            unit="s"
        )

        interaction_rows.append({
            "user_id": user_id,
            "product_id": product_id,
            "event_type": event_type,
            "timestamp": timestamp,
        })

interactions_df = pd.DataFrame(interaction_rows)

# Sort chronologically for time-based train/test splitting.
interactions_df = interactions_df.sort_values(
    "timestamp"
).reset_index(drop=True)

# --------------------------------------------------
# DATA VALIDATION
# --------------------------------------------------

assert len(products_df) == 36
assert len(users_df) == 500
assert len(interactions_df) == 15000

assert products_df["product_id"].is_unique
assert users_df["user_id"].is_unique

assert not interactions_df.duplicated(
    subset=["user_id", "product_id"]
).any()

assert set(interactions_df["product_id"]).issubset(
    set(products_df["product_id"])
)

assert set(interactions_df["user_id"]).issubset(
    set(users_df["user_id"])
)

interactions_per_user = interactions_df.groupby(
    "user_id"
).size()

assert (interactions_per_user == 30).all()

# --------------------------------------------------
# SAVE CSV FILES
# --------------------------------------------------

products_path = os.path.join(DATASET_DIR, "products.csv")
users_path = os.path.join(DATASET_DIR, "users.csv")
interactions_path = os.path.join(
    DATASET_DIR, "interactions.csv"
)

products_df.to_csv(products_path, index=False)
users_df.to_csv(users_path, index=False)
interactions_df.to_csv(interactions_path, index=False)

# --------------------------------------------------
# DISPLAY GENERATION SUMMARY
# --------------------------------------------------

print("\n" + "=" * 55)
print("       E-COMMERCE DATA GENERATION SUCCESSFUL")
print("=" * 55)

print("\nDATASET SUMMARY")
print("-" * 55)
print("Total Products       :", len(products_df))
print("Total Users          :", len(users_df))
print("Total Interactions   :", len(interactions_df))

print("\nPRODUCT CATEGORY COUNTS")
print("-" * 55)
print(products_df["category"].value_counts().to_string())

print("\nEVENT TYPE COUNTS")
print("-" * 55)
print(interactions_df["event_type"].value_counts().to_string())

print("\nPRODUCT CATALOGUE SAMPLE")
print("-" * 55)
print(
    products_df[
        ["product_id", "product_name", "brand", "category", "price"]
    ].head(10).to_string(index=False)
)

print("\nUSER SAMPLE")
print("-" * 55)
print(users_df.head(5).to_string(index=False))

print("\nFILES SAVED")
print("-" * 55)
print(products_path)
print(users_path)
print(interactions_path)

print("\nAll validation checks passed!")
print("=" * 55)
