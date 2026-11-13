

\# 🛍️ ShopSmart AI – E-Commerce Product Recommendation Engine



\*\*Organization:\*\* ALGIONS AI LABS PRIVATE LIMITED  

\*\*Project:\*\* E-Commerce Product Recommendation Engine  

\*\*Technology:\*\* Python, Machine Learning and Streamlit



\## 1. Project Overview



ShopSmart AI is an AI-powered e-commerce product recommendation system that suggests relevant products to customers based on their previous interactions and overall product popularity.



The application provides an interactive shopping experience through a Streamlit web interface.



\## 2. Project Objectives



\- Build a Top-N product recommendation engine.

\- Implement and compare two recommendation approaches.

\- Provide personalized product recommendations.

\- Handle new visitors using cold-start recommendations.

\- Evaluate recommendation quality using standard metrics.

\- Develop an interactive web application.



\## 3. Dataset Description



The project uses a synthetic e-commerce dataset created for demonstration and academic purposes.



| Dataset | Records |

|---|---:|

| Products | 36 |

| Users | 500 |

| Interactions | 15,000 |



\### Dataset Files



\- `Dataset/products.csv`

\- `Dataset/users.csv`

\- `Dataset/interactions.csv`



The interaction dataset contains three event types:



\- View

\- Cart

\- Purchase



Interaction timestamps are used for chronological training and testing.



\## 4. Technologies Used



\- Python

\- Pandas

\- NumPy

\- Scikit-learn

\- SciPy

\- Streamlit

\- Matplotlib

\- Seaborn

\- Joblib

\- Pytest



\## 5. Recommendation Models



\### A. Popularity-Based Recommendation



This model recommends products according to their overall interaction strength.



It serves as the baseline recommendation approach and supports new visitors.



\### B. Item-Item Collaborative Filtering



This model identifies similarities between products based on user interaction patterns.



It uses a customer's interaction history to recommend products similar to those they previously interacted with.



\### Cold-Start Handling



When a new visitor has no interaction history, the system uses popularity-based recommendations.



\## 6. Data Preprocessing



The project includes:



\- Data validation.

\- Handling missing values.

\- Interaction processing.

\- Product and user ID validation.

\- Interaction weighting.

\- Chronological train-test splitting.

\- Prevention of future interaction leakage.



Interaction weights:



| Event | Weight |

|---|---:|

| View | 1 |

| Cart | 3 |

| Purchase | 5 |



\## 7. Model Evaluation



The models are evaluated using:



\- Precision@10

\- Recall@10

\- NDCG@10

\- Catalog Coverage



\### Model Comparison



| Metric | Popularity Baseline | Collaborative Filtering |

|---|---:|---:|

| Precision@10 | 14.27% | 14.46% |

| Recall@10 | 80.17% | 81.62% |

| NDCG@10 | 41.55% | 42.89% |

| Catalog Coverage | 100% | 100% |



Collaborative Filtering performs slightly better than the popularity baseline across Precision@10, Recall@10 and NDCG@10 on the held-out evaluation data.



\## 8. Project Structure



```text

Ecommerce\_Recommendation\_Project/

│

├── app.py

├── Dataset/

│   ├── products.csv

│   ├── users.csv

│   └── interactions.csv

│

├── Models/

│   └── recommendation\_model.pkl

│

├── Notebooks/

│

├── assets/

│   └── products/

│       ├── PRD001.jpg

│       ├── ...

│       └── PRD036.jpg

│

├── src/

│   ├── generate\_data.py

│   ├── eda.py

│   ├── preprocessing.py

│   ├── train\_models.py

│   └── generate\_product\_images.py

│

├── reports/

│   └── model\_comparison.csv

│

├── screenshots/

│

├── tests/

│   └── test\_recommendations.py

│

└── requirements.txt

```



\## 9. Installation and Execution



\### Step 1: Clone or extract the project



Open the project folder in Command Prompt.



\### Step 2: Create a virtual environment



```bash

python -m venv venv

```



\### Step 3: Activate the environment



```bash

venv\\Scripts\\activate

```



\### Step 4: Install dependencies



```bash

pip install -r requirements.txt

```



\### Step 5: Run the Streamlit application



```bash

streamlit run app.py

```



\### Step 6: Run automated tests



```bash

python -m pytest tests\\test\_recommendations.py -v

```



\### Optional: Rebuild the dataset and model



For a fresh project setup, generate the dataset first:



```bash

python src\\generate\_data.py

```



Then train and save the recommendation models:



```bash

python src\\train\_models.py

```



The application loads the saved model from:



```text

Models/recommendation\_model.pkl

```



\## 10. Testing and Validation



Automated tests verify:



1\. Saved model loading.

2\. Product catalogue integrity.

3\. User data integrity.

4\. Popularity recommendations.

5\. Collaborative Filtering recommendations.

6\. Cold-start recommendations.

7\. Exclusion of previously interacted products.

8\. Availability of product images.

9\. Evaluation metric availability.



\*\*Latest test result: 9 out of 9 tests passed.\*\*



\## 11. Application Features



\- Interactive customer selection.

\- Two recommendation approaches.

\- Adjustable Top-N recommendations.

\- Product photographs.

\- Product categories and prices.

\- Customer interaction history.

\- Model performance comparison.

\- Complete product catalogue.

\- Cold-start recommendation support.



\## 12. Responsible Use and Limitations



This project uses synthetic user, product and interaction data.



The recommendations are intended for demonstration and learning purposes. They are not verified predictions of real customer purchases.



The model may be limited by the small product catalogue, synthetic interaction patterns and lack of real-time customer feedback.



\## 13. Conclusion



ShopSmart AI demonstrates how machine learning can be used to build an e-commerce product recommendation engine.



The project compares popularity-based recommendations with Item-Item Collaborative Filtering and evaluates their performance using ranking metrics.



The Streamlit application makes the recommendation system interactive and accessible.



The project provides a foundation for future improvements using real-world product catalogues, customer feedback and larger interaction datasets.



