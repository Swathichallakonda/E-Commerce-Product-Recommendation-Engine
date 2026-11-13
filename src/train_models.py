
import os
import pickle
import numpy as np
import pandas as pd

from collections import defaultdict
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

RANDOM_SEED = 42
TOP_K = 10

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "Dataset")
MODEL_DIR = os.path.join(BASE_DIR, "Models")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

np.random.seed(RANDOM_SEED)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("\nLoading datasets...")

products = pd.read_csv(
    os.path.join(DATASET_DIR, "products.csv")
)

users = pd.read_csv(
    os.path.join(DATASET_DIR, "users.csv")
)

interactions = pd.read_csv(
    os.path.join(DATASET_DIR, "interactions.csv")
)

interactions["timestamp"] = pd.to_datetime(
    interactions["timestamp"],
    errors="coerce"
)

interactions = interactions.dropna(
    subset=["user_id", "product_id", "event_type", "timestamp"]
)

print("Products:", len(products))
print("Users:", len(users))
print("Interactions:", len(interactions))

# --------------------------------------------------
# DATA VALIDATION
# --------------------------------------------------

required_interaction_columns = {
    "user_id", "product_id", "event_type", "timestamp"
}

assert required_interaction_columns.issubset(
    interactions.columns
)

assert products["product_id"].is_unique
assert users["user_id"].is_unique

assert set(interactions["product_id"]).issubset(
    set(products["product_id"])
)

assert set(interactions["user_id"]).issubset(
    set(users["user_id"])
)

# --------------------------------------------------
# FEATURE ENGINEERING
# --------------------------------------------------

event_weights = {
    "view": 1,
    "cart": 3,
    "purchase": 5
}

interactions["interaction_weight"] = (
    interactions["event_type"].map(event_weights)
)

interactions = interactions.dropna(
    subset=["interaction_weight"]
)

interactions["interaction_weight"] = (
    interactions["interaction_weight"].astype(float)
)

# --------------------------------------------------
# CHRONOLOGICAL TRAIN-TEST SPLIT
# --------------------------------------------------

interactions = interactions.sort_values(
    "timestamp"
).reset_index(drop=True)

split_index = int(len(interactions) * 0.80)

train_data = interactions.iloc[:split_index].copy()
test_data = interactions.iloc[split_index:].copy()

print("\nCHRONOLOGICAL DATA SPLIT")
print("-" * 45)
print("Training interactions:", len(train_data))
print("Testing interactions :", len(test_data))
print(
    "Training end date    :",
    train_data["timestamp"].max()
)
print(
    "Testing start date   :",
    test_data["timestamp"].min()
)

assert train_data["timestamp"].max() <= (
    test_data["timestamp"].min()
)

# --------------------------------------------------
# BUILD USER HISTORY
# --------------------------------------------------

user_history = defaultdict(dict)

for row in train_data.itertuples(index=False):

    user_id = row.user_id
    product_id = row.product_id
    weight = row.interaction_weight

    user_history[user_id][product_id] = (
        user_history[user_id].get(product_id, 0) + weight
    )

user_history = dict(user_history)

# --------------------------------------------------
# BUILD POPULARITY MODEL
# --------------------------------------------------

product_popularity = (
    train_data.groupby("product_id")["interaction_weight"]
    .sum()
    .sort_values(ascending=False)
)

popularity_ranking = product_popularity.index.tolist()

# Include products without training interactions at the end.
for product_id in products["product_id"]:
    if product_id not in popularity_ranking:
        popularity_ranking.append(product_id)

print("\nPopularity model created.")

# --------------------------------------------------
# BUILD ITEM-ITEM COLLABORATIVE FILTERING
# --------------------------------------------------

print("\nBuilding collaborative filtering model...")

all_users = sorted(train_data["user_id"].unique())
all_products = products["product_id"].tolist()

user_index = {
    user_id: index
    for index, user_id in enumerate(all_users)
}

product_index = {
    product_id: index
    for index, product_id in enumerate(all_products)
}

row_indices = []
column_indices = []
values = []

for row in train_data.itertuples(index=False):

    row_indices.append(user_index[row.user_id])
    column_indices.append(product_index[row.product_id])
    values.append(row.interaction_weight)

user_item_matrix = csr_matrix(
    (values, (row_indices, column_indices)),
    shape=(len(all_users), len(all_products))
)

similarity_matrix = cosine_similarity(
    user_item_matrix.T,
    dense_output=True
)

item_similarity = {}

for i, product_id in enumerate(all_products):

    similarities = {}

    for j, other_product_id in enumerate(all_products):

        if i != j and similarity_matrix[i, j] > 0:
            similarities[other_product_id] = float(
                similarity_matrix[i, j]
            )

    item_similarity[product_id] = similarities

print("Collaborative filtering model created.")

# --------------------------------------------------
# RECOMMENDATION FUNCTIONS
# --------------------------------------------------

def popularity_recommend(user_id, top_k=10):

    already_seen = set(
        user_history.get(user_id, {}).keys()
    )

    recommendations = []

    for product_id in popularity_ranking:

        if product_id not in already_seen:
            recommendations.append(product_id)

        if len(recommendations) == top_k:
            break

    return recommendations


def collaborative_recommend(user_id, top_k=10):

    history = user_history.get(user_id, {})

    # Cold-start fallback.
    if not history:
        return popularity_recommend(user_id, top_k)

    already_seen = set(history.keys())
    scores = defaultdict(float)

    for interacted_product, interaction_weight in history.items():

        similar_products = item_similarity.get(
            interacted_product, {}
        )

        for recommended_product, similarity in (
            similar_products.items()
        ):

            if recommended_product not in already_seen:

                scores[recommended_product] += (
                    interaction_weight * similarity
                )

    # Sort by recommendation score.
    ranked_products = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    recommendations = [
        product_id
        for product_id, score in ranked_products
    ]

    # Fill remaining positions using popularity.
    if len(recommendations) < top_k:

        for product_id in popularity_ranking:

            if (
                product_id not in already_seen
                and product_id not in recommendations
            ):
                recommendations.append(product_id)

            if len(recommendations) >= top_k:
                break

    return recommendations[:top_k]


# --------------------------------------------------
# PREPARE HELD-OUT PURCHASES
# --------------------------------------------------

test_purchases = test_data[
    test_data["event_type"] == "purchase"
]

actual_purchases = (
    test_purchases.groupby("user_id")["product_id"]
    .apply(set)
    .to_dict()
)

# Evaluate only purchases of products not already seen
# by that user in the training period.

eligible_actual = {}

for user_id, purchased_products in actual_purchases.items():

    previously_seen = set(
        user_history.get(user_id, {}).keys()
    )

    relevant_products = (
        purchased_products - previously_seen
    )

    if relevant_products:
        eligible_actual[user_id] = relevant_products

print("\nEVALUATION USERS")
print("-" * 45)
print("Users with eligible unseen purchases:", len(eligible_actual))

# --------------------------------------------------
# EVALUATION METRICS
# --------------------------------------------------

def calculate_metrics(recommendation_function, model_name):

    precision_values = []
    recall_values = []
    ndcg_values = []

    recommended_catalogue = set()

    for user_id, relevant_products in eligible_actual.items():

        recommendations = recommendation_function(
            user_id, TOP_K
        )

        recommended_catalogue.update(recommendations)

        hits = [
            1 if product_id in relevant_products else 0
            for product_id in recommendations
        ]

        hit_count = sum(hits)

        precision = hit_count / TOP_K

        recall = hit_count / len(relevant_products)

        dcg = sum(
            hit / np.log2(rank + 2)
            for rank, hit in enumerate(hits)
        )

        ideal_hits = min(
            len(relevant_products), TOP_K
        )

        idcg = sum(
            1 / np.log2(rank + 2)
            for rank in range(ideal_hits)
        )

        ndcg = dcg / idcg if idcg > 0 else 0

        precision_values.append(precision)
        recall_values.append(recall)
        ndcg_values.append(ndcg)

    if not eligible_actual:
        return {
            "Model": model_name,
            "Precision@10": 0.0,
            "Recall@10": 0.0,
            "NDCG@10": 0.0,
            "Catalog Coverage": 0.0
        }

    return {
        "Model": model_name,
        "Precision@10": float(np.mean(precision_values)),
        "Recall@10": float(np.mean(recall_values)),
        "NDCG@10": float(np.mean(ndcg_values)),
        "Catalog Coverage": (
            len(recommended_catalogue) / len(all_products)
        )
    }


# --------------------------------------------------
# COMPARE MODELS
# --------------------------------------------------

print("\nEvaluating popularity baseline...")

popularity_metrics = calculate_metrics(
    popularity_recommend,
    "Popularity Baseline"
)

print("\nEvaluating collaborative filtering...")

collaborative_metrics = calculate_metrics(
    collaborative_recommend,
    "Collaborative Filtering"
)

comparison = pd.DataFrame([
    popularity_metrics,
    collaborative_metrics
])

comparison.to_csv(
    os.path.join(REPORT_DIR, "model_comparison.csv"),
    index=False
)

print("\n" + "=" * 65)
print("                 MODEL COMPARISON")
print("=" * 65)

print(comparison.to_string(index=False))

# --------------------------------------------------
# SAVE MODEL ARTIFACT
# --------------------------------------------------

model_artifact = {
    "user_history": user_history,
    "popularity_ranking": popularity_ranking,
    "item_similarity": item_similarity,
    "products": products,
    "users": users,
    "metrics": comparison,
    "event_weights": event_weights,
    "top_k": TOP_K,
    "training_interactions": len(train_data),
    "testing_interactions": len(test_data),
    "eligible_evaluation_users": len(eligible_actual),
    "random_seed": RANDOM_SEED
}

model_path = os.path.join(
    MODEL_DIR,
    "recommendation_model.pkl"
)

with open(model_path, "wb") as file:
    pickle.dump(model_artifact, file)

# --------------------------------------------------
# SAMPLE RECOMMENDATIONS
# --------------------------------------------------

sample_user = next(iter(user_history))

sample_recommendations = collaborative_recommend(
    sample_user, TOP_K
)

product_lookup = products.set_index("product_id")

print("\nSAMPLE RECOMMENDATIONS")
print("-" * 65)
print("User ID:", sample_user)

for rank, product_id in enumerate(
    sample_recommendations, start=1
):

    product = product_lookup.loc[product_id]

    print(
        f"{rank}. {product['product_name']} | "
        f"{product['brand']} | "
        f"{product['category']} | "
        f"Rs. {product['price']}"
    )

print("\nMODEL SAVED SUCCESSFULLY")
print(model_path)

print("\nEvaluation report saved:")
print(
    os.path.join(
        REPORT_DIR,
        "model_comparison.csv"
    )
)

print("\nTraining and evaluation completed!")
