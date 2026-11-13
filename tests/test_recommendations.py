
import os
import pickle
from collections import defaultdict

import pytest


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "Models",
    "recommendation_model.pkl"
)

IMAGE_FOLDER = os.path.join(
    BASE_DIR,
    "assets",
    "products"
)


# --------------------------------------------------
# LOAD SAVED MODEL
# --------------------------------------------------

@pytest.fixture(scope="module")
def artifact():

    assert os.path.exists(MODEL_PATH), (
        "Saved recommendation model was not found."
    )

    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    return model


# --------------------------------------------------
# RECOMMENDATION FUNCTIONS
# --------------------------------------------------

def popularity_recommend(artifact, user_id=None, top_k=10):

    user_history = artifact["user_history"]
    popularity_ranking = artifact["popularity_ranking"]

    already_seen = set(
        user_history.get(user_id, {}).keys()
    ) if user_id else set()

    recommendations = []

    for product_id in popularity_ranking:

        if product_id not in already_seen:
            recommendations.append(product_id)

        if len(recommendations) >= top_k:
            break

    return recommendations


def collaborative_recommend(artifact, user_id, top_k=10):

    user_history = artifact["user_history"]
    item_similarity = artifact["item_similarity"]
    popularity_ranking = artifact["popularity_ranking"]

    history = user_history.get(user_id, {})

    if not history:
        return popularity_recommend(
            artifact,
            user_id,
            top_k
        )

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

    ranked_products = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    recommendations = [
        product_id
        for product_id, score in ranked_products
    ]

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
# TEST 1: MODEL LOADING
# --------------------------------------------------

def test_saved_model_loads(artifact):

    assert "user_history" in artifact
    assert "popularity_ranking" in artifact
    assert "item_similarity" in artifact
    assert "products" in artifact
    assert "users" in artifact
    assert "metrics" in artifact


# --------------------------------------------------
# TEST 2: PRODUCT CATALOGUE
# --------------------------------------------------

def test_product_catalogue(artifact):

    products = artifact["products"]

    assert len(products) == 36
    assert products["product_id"].nunique() == 36


# --------------------------------------------------
# TEST 3: USER DATA
# --------------------------------------------------

def test_user_data(artifact):

    users = artifact["users"]

    assert len(users) == 500
    assert users["user_id"].nunique() == 500


# --------------------------------------------------
# TEST 4: POPULARITY RECOMMENDATIONS
# --------------------------------------------------

def test_popularity_recommendations(artifact):

    recommendations = popularity_recommend(
        artifact,
        None,
        10
    )

    valid_products = set(
        artifact["products"]["product_id"]
    )

    assert len(recommendations) == 10
    assert len(set(recommendations)) == 10
    assert set(recommendations).issubset(valid_products)


# --------------------------------------------------
# TEST 5: COLLABORATIVE FILTERING
# --------------------------------------------------

def test_collaborative_recommendations(artifact):

    user_id = next(iter(artifact["user_history"]))

    recommendations = collaborative_recommend(
        artifact,
        user_id,
        10
    )

    valid_products = set(
        artifact["products"]["product_id"]
    )

    already_seen = set(
        artifact["user_history"].get(user_id, {}).keys()
    )

    assert len(recommendations) == 10
    assert len(set(recommendations)) == 10
    assert set(recommendations).issubset(valid_products)
    assert set(recommendations).isdisjoint(already_seen)


# --------------------------------------------------
# TEST 6: COLD-START RECOMMENDATIONS
# --------------------------------------------------

def test_cold_start_recommendations(artifact):

    recommendations = popularity_recommend(
        artifact,
        None,
        10
    )

    assert len(recommendations) == 10
    assert len(set(recommendations)) == 10


# --------------------------------------------------
# TEST 7: PREVIOUSLY SEEN PRODUCTS EXCLUDED
# --------------------------------------------------

def test_seen_products_are_excluded(artifact):

    user_id = next(iter(artifact["user_history"]))

    already_seen = set(
        artifact["user_history"][user_id].keys()
    )

    recommendations = popularity_recommend(
        artifact,
        user_id,
        10
    )

    assert set(recommendations).isdisjoint(already_seen)


# --------------------------------------------------
# TEST 8: PRODUCT IMAGES
# --------------------------------------------------

def test_all_product_images_exist(artifact):

    products = artifact["products"]

    missing_images = []

    for product_id in products["product_id"]:

        image_path = os.path.join(
            IMAGE_FOLDER,
            f"{product_id}.jpg"
        )

        if not os.path.isfile(image_path):
            missing_images.append(product_id)

    assert not missing_images, (
        f"Missing product images: {missing_images}"
    )


# --------------------------------------------------
# TEST 9: MODEL EVALUATION METRICS
# --------------------------------------------------

def test_model_metrics(artifact):

    metrics = artifact["metrics"]

    assert len(metrics) >= 2

    required_columns = [
        "Model",
        "Precision@10",
        "Recall@10",
        "NDCG@10",
        "Catalog Coverage"
    ]

    for column in required_columns:
        assert column in metrics.columns
