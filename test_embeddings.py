from embeddings import get_embeddings


embeddings = get_embeddings()

text = "Artificial Intelligence is a branch of computer science."

vector = embeddings.embed_query(text)

print("Embedding generated successfully!")
print("Vector dimensions:", len(vector))
print("First 10 values:", vector[:10])