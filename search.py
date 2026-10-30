import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

documents = [
    "Veritabanı indeksleri, diskteki arama süresini ciddi oranda azaltır.",
    "B-Tree ve HNSW algoritmaları arama problemlerinde yaygın kullanılır.",
    "Akşam yemeğinde makarna ve domates soslu tavuk yaptık.",
    "Python ve C dilleri ile sistem programlama yapılabilir.",
    "Kediler günün büyük bir bölümünü uyuyarak geçirirler."
]

print("Dokümanlar vektörleştiriliyor...")
doc_vectors = model.encode(documents)

def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return dot_product / (norm_v1 * norm_v2)

print("\n" + "=" * 50)
print("Arama motoru hazır! Çıkmak için 'q' yazıp Enter'a basabilirsin.")
print("=" * 50)

while True:
    query = input("\nAramak istediğiniz cümle: ").strip()
    
    if query.lower() == 'q' or query == '':
        print("Arama motorundan çıkılıyor.")
        break

    query_vector = model.encode(query)

    scores = []
    for idx, doc_vec in enumerate(doc_vectors):
        sim = cosine_similarity(query_vector, doc_vec)
        scores.append((sim, documents[idx]))

    scores.sort(key=lambda x: x[0], reverse=True)

    print(f"\nSonuçlar ('{query}'):")
    print("-" * 40)
    for score, doc in scores:
        print(f"[{score:.4f}] {doc}")
