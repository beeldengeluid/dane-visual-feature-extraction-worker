import numpy as np

dim = 512
db_size = 1e6

# Step 1: Initialize the query vector, here we simply use a random vector as the query
query = np.random.rand(1, dim).astype(np.float32)

# Step 2: Initialize the database vectors, here we simply use vectors as the database
db = np.random.rand(int(db_size), dim).astype(np.float32)

# Step 3: Compute the cosine similarity between the query and database vectors
# Note: the cosine similarity is computed by the dot product of the normalized query and database vectors
query = query / np.linalg.norm(query)
db = db / np.linalg.norm(db, axis=1, keepdims=True)
sim = np.dot(query, db.T).squeeze(0)

# Step 4: Sort the similarity scores in descending order
sorted_idx = np.argsort(sim)[::-1]

# Step 5: Print the top-10 most similar vectors
print("Top-10 most similar vectors:", sorted_idx[:10])
