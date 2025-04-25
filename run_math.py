import os
import logging
from litepolis_math import fetch_r_matrix
from litepolis_math.algorithms import PCA
from litepolis_math.algorithms import KMeans
from litepolis_math.validation import validate_matrix

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Step 1: Connect and build R matrix
try:
    r_matrix = fetch_r_matrix()
    logging.info(f"Fetched R matrix with shape: {r_matrix.shape}")

    validate_matrix(r_matrix)  # Ensure data is clean
    logging.info("R matrix validated.")

    # Step 2: Apply PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(r_matrix.values)
    logging.info(f"PCA applied. Result shape: {pca_result.shape}")

    # Step 3: Cluster users
    kmeans = KMeans(n_clusters=3)
    user_clusters = kmeans.fit_predict(pca_result)
    logging.info("KMeans clustering applied.")

    print("User clusters:", user_clusters)

except Exception as e:
    logging.error(f"An error occurred: {e}")