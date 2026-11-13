
import os
import pickle
from collections import defaultdict

import pandas as pd
import streamlit as st


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="ShopSmart AI | Product Recommendations",
    page_icon="🛍️",
    layout="wide"
)


# --------------------------------------------------
# CUSTOM STYLING
# --------------------------------------------------

st.markdown("""
<style>
.main {
    background-color: #f8f9fc;
}

.main-title {
    font-size: 38px;
    font-weight: 800;
    color: #24243e;
}

.subtitle {
    color: #686880;
    font-size: 16px;
}

.product-card {
    background: white;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #e8e8ef;
    margin-bottom: 15px;
}

.product-name {
    font-size: 17px;
    font-weight: 700;
    color: #252542;
}

.product-brand {
    color: #77778a;
    font-size: 13px;
}

.product-price {
    font-size: 21px;
    font-weight: 800;
    color: #176b50;
}

.small-note {
    color: #77778a;
    font-size: 12px;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD SAVED MODEL
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "Models",
    "recommendation_model.pkl"
)

# Product image folder
IMAGE_FOLDER = os.path.join(
    BASE_DIR,
    "assets",
    "products"
)


if not os.path.exists(MODEL_PATH):
    st.error(
        "Trained model not found. "
        "Please run python src\\train_models.py first."
    )
    st.stop()


@st.cache_resource
def load_model(path):
    with open(path, "rb") as file:
        return pickle.load(file)


artifact = load_model(MODEL_PATH)

user_history = artifact["user_history"]
popularity_ranking = artifact["popularity_ranking"]
item_similarity = artifact["item_similarity"]

products = artifact["products"].copy()
users = artifact["users"].copy()
metrics = artifact["metrics"].copy()

products["price"] = pd.to_numeric(
    products["price"],
    errors="coerce"
)

product_lookup = products.set_index("product_id")


# --------------------------------------------------
# PRODUCT IMAGE FUNCTION
# --------------------------------------------------

def get_product_image(product_id):

    image_path = os.path.join(
        IMAGE_FOLDER,
        f"{product_id}.jpg"
    )

    if os.path.isfile(image_path):
        return image_path

    return None


# --------------------------------------------------
# CATEGORY ICONS
# --------------------------------------------------

category_icons = {
    "Footwear": "👟",
    "Men's Clothing": "👕",
    "Women's Clothing": "👗",
    "Accessories": "⌚",
    "Electronics": "🎧",
    "Beauty": "💄",
    "Home & Kitchen": "🏠",
    "Sports": "🏸"
}


# --------------------------------------------------
# RECOMMENDATION FUNCTIONS
# --------------------------------------------------

def popularity_recommend(user_id=None, top_k=10):

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


def collaborative_recommend(user_id, top_k=10):

    history = user_history.get(user_id, {})

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
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🛍️ ShopSmart AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Discover products selected using Artificial Intelligence '
    'and Machine Learning.'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("⚙️ Recommendation Settings")

st.sidebar.markdown("### Select Customer")

user_options = ["New Visitor (Cold Start)"]

user_map = {}

for row in users.itertuples(index=False):

    label = f"{row.user_name} ({row.user_id})"

    user_options.append(label)
    user_map[label] = row.user_id

selected_user_label = st.sidebar.selectbox(
    "Choose a user",
    user_options
)

if selected_user_label == "New Visitor (Cold Start)":
    selected_user = None
else:
    selected_user = user_map[selected_user_label]


st.sidebar.markdown("### Select Recommendation Model")

model_choice = st.sidebar.radio(
    "Recommendation approach",
    [
        "Collaborative Filtering",
        "Popularity Baseline"
    ]
)

top_k = st.sidebar.slider(
    "Number of recommendations",
    min_value=5,
    max_value=20,
    value=10
)

st.sidebar.divider()

st.sidebar.info(
    "Collaborative Filtering uses previous user-product "
    "interactions. Popularity Baseline recommends products "
    "ranked by overall interaction strength."
)


# --------------------------------------------------
# MAIN METRICS
# --------------------------------------------------

st.subheader("📊 Project Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Products", len(products))
col2.metric("Customers", len(users))
col3.metric(
    "Interactions",
    f"{artifact['training_interactions'] + artifact['testing_interactions']:,}"
)
col4.metric(
    "Categories",
    products["category"].nunique()
)

st.divider()


# --------------------------------------------------
# GENERATE RECOMMENDATIONS
# --------------------------------------------------

st.subheader("✨ Recommended Products")

if selected_user is None:

    st.info(
        "You are browsing as a new visitor. "
        "Showing popular products as a cold-start recommendation."
    )

    recommendation_ids = popularity_recommend(
        None,
        top_k
    )

    recommendation_reason = (
        "Recommended using overall product popularity."
    )

else:

    selected_user_data = users[
        users["user_id"] == selected_user
    ].iloc[0]

    st.write(
        f"Welcome, **{selected_user_data['user_name']}!**"
    )

    st.caption(
        f"Preferred category: "
        f"{selected_user_data['preferred_category']} "
        f"| Age group: {selected_user_data['age_group']}"
    )

    if model_choice == "Collaborative Filtering":

        recommendation_ids = collaborative_recommend(
            selected_user,
            top_k
        )

        recommendation_reason = (
            "Recommended using item-item collaborative "
            "filtering and your previous interactions."
        )

    else:

        recommendation_ids = popularity_recommend(
            selected_user,
            top_k
        )

        recommendation_reason = (
            "Recommended using overall product popularity, "
            "excluding products already interacted with."
        )


st.caption(recommendation_reason)


# --------------------------------------------------
# DISPLAY PRODUCT CARDS
# --------------------------------------------------

if not recommendation_ids:

    st.warning("No recommendations available.")

else:

    for start in range(0, len(recommendation_ids), 2):

        product_columns = st.columns(2)

        for offset, column in enumerate(product_columns):

            index = start + offset

            if index >= len(recommendation_ids):
                break

            product_id = recommendation_ids[index]

            if product_id not in product_lookup.index:
                continue

            product = product_lookup.loc[product_id]

            with column:

                with st.container(border=True):

                    # Product photograph
                    image_path = get_product_image(product_id)

                    if image_path:
                        st.image(
                            image_path,
                            use_container_width=True
                        )
                    else:
                        st.caption("Product image not available.")

                    icon = category_icons.get(
                        product["category"],
                        "🛍️"
                    )

                    st.markdown(
                        f"### {icon} {product['product_name']}"
                    )

                    st.caption(
                        f"Brand: {product['brand']}"
                    )

                    st.markdown(
                        f"**Category:** {product['category']}"
                    )

                    st.markdown(
                        f"**Type:** {product['subcategory']}"
                    )

                    st.markdown(
                        f'<div class="product-price">'
                        f'₹{product["price"]:,.0f}'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                    st.caption(
                        f"Product ID: {product_id}"
                    )


# --------------------------------------------------
# USER INTERACTION HISTORY
# --------------------------------------------------

if selected_user is not None:

    st.divider()

    st.subheader("🧾 Customer Interaction History")

    history = user_history.get(selected_user, {})

    if history:

        history_rows = []

        for product_id, weight in history.items():

            if product_id in product_lookup.index:

                product = product_lookup.loc[product_id]

                history_rows.append({
                    "Product": product["product_name"],
                    "Brand": product["brand"],
                    "Category": product["category"],
                    "Price": f"₹{product['price']:,.0f}",
                    "Interaction Weight": weight
                })

        history_df = pd.DataFrame(history_rows)

        st.write(
            f"Total previously interacted products: "
            f"**{len(history_df)}**"
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No previous interaction history. "
            "Popularity recommendations are used."
        )


# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

st.divider()

st.subheader("📈 Model Performance Comparison")

display_metrics = metrics.copy()

metric_columns = [
    "Precision@10",
    "Recall@10",
    "NDCG@10",
    "Catalog Coverage"
]

for column in metric_columns:
    display_metrics[column] = (
        display_metrics[column] * 100
    ).round(2)

st.caption(
    "Evaluation on held-out future purchase interactions. "
    "Values below are percentages."
)

st.dataframe(
    display_metrics,
    use_container_width=True,
    hide_index=True
)

chart_data = metrics.set_index("Model")[
    [
        "Precision@10",
        "Recall@10",
        "NDCG@10",
        "Catalog Coverage"
    ]
]

st.bar_chart(chart_data)


# --------------------------------------------------
# COMPLETE PRODUCT CATALOGUE
# --------------------------------------------------

st.divider()

st.subheader("🛒 Explore Product Catalogue")

available_categories = [
    "All Categories"
] + sorted(products["category"].unique().tolist())

selected_category = st.selectbox(
    "Filter by category",
    available_categories
)

if selected_category == "All Categories":

    filtered_products = products.copy()

else:

    filtered_products = products[
        products["category"] == selected_category
    ].copy()

st.write(
    f"Showing **{len(filtered_products)}** products"
)

# Display catalogue products with matching photographs
for start in range(0, len(filtered_products), 3):

    catalogue_columns = st.columns(3)

    for offset, column in enumerate(catalogue_columns):

        index = start + offset

        if index >= len(filtered_products):
            break

        product = filtered_products.iloc[index]

        with column:

            with st.container(border=True):

                image_path = get_product_image(
                    product["product_id"]
                )

                if image_path:
                    st.image(
                        image_path,
                        use_container_width=True
                    )
                else:
                    st.caption("Product image not available.")

                icon = category_icons.get(
                    product["category"],
                    "🛍️"
                )

                st.markdown(
                    f"**{icon} {product['product_name']}**"
                )

                st.caption(
                    f"Brand: {product['brand']}"
                )

                st.caption(
                    f"Category: {product['category']}"
                )

                st.caption(
                    f"Subcategory: {product['subcategory']}"
                )

                st.markdown(
                    f'<div class="product-price">'
                    f'₹{product["price"]:,.0f}'
                    f'</div>',
                    unsafe_allow_html=True
                )

                st.caption(
                    f"Product ID: {product['product_id']}"
                )


# --------------------------------------------------
# ABOUT THE PROJECT
# --------------------------------------------------

st.divider()

with st.expander("ℹ️ About ShopSmart AI"):

    st.markdown("""
    **Project:** E-Commerce Product Recommendation Engine

    **Organization:** ALGIONS AI LABS PRIVATE LIMITED

    **Tech Stack:**
    - Python
    - Pandas and NumPy
    - Scikit-learn
    - SciPy
    - Streamlit

    **Recommendation Models:**

    1. Popularity-Based Recommendation
    2. Item-Item Collaborative Filtering

    **Dataset:**
    - 500 synthetic demo users
    - 36 demo catalogue products
    - 15,000 synthetic interactions

    **Evaluation Metrics:**
    - Precision@10
    - Recall@10
    - NDCG@10
    - Catalog Coverage

    **Cold Start:** New visitors receive popularity-based
    recommendations.

    **Limitations:** This is a demonstration system using
    synthetic interaction data. Its recommendations are not
    verified predictions of actual customer purchases.
    """)

st.caption(
    "ShopSmart AI | E-Commerce Recommendation Engine | "
    "Academic and internship demonstration project"
)
