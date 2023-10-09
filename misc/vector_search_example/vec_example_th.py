import torch

dim = 512
db_size = 1e6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 1: Initialize the query vector, here we simply use a random vector as the query
query = torch.randn(1, dim).to(device)

# Step 2: Initialize the database vectors, here we simply use vectors as the database
db = torch.randn(int(db_size), dim).to(device)

# Step 3: Compute the cosine similarity between the query and database vectors
# Note: the cosine similarity is computed by the dot product of the normalized query and database vectors
query_norm = query / torch.norm(query)
db_norm = db / torch.norm(db, dim=1, keepdim=True)
sim = query_norm @ db_norm.T

# Step 4: Sort the similarity scores in descending order
sorted_idx = sim.argsort(descending=True)

# Step 5: Print the top-10 most similar vectors
print("Top-10 most similar vectors:", sorted_idx[:10])
