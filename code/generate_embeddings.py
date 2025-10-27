"""
Generate cell and phenotype embeddings using OpenAI API
"""

import numpy as np
from openai import OpenAI

# API key
api_key = "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"

client = OpenAI(api_key=api_key)

# Exact strings from benchmark
cells = [
    'NG2-3112 mouse glioblastoma cells',
    '3LL Lewis lung carcinoma cells'
]

phenotypes = [
    'increased sensitivity to gliocidin and subsequently glioblastoma cell death',
    'increased resistance to PD1 blockade and lung carcinoma cell survival',
    'decreased sensitivity to gliocidin and subsequently glioblastoma cell survival',
    'decreased resistance to PD1 blockade and lung carcinoma cell death'
]

print("="*70)
print("GENERATING BENCHMARK EMBEDDINGS")
print("="*70)

# Generate cell embeddings
print("\n[1/2] Cell embeddings...")
cell_embeddings = {}
for i, cell in enumerate(cells, 1):
    print(f"  [{i}/{len(cells)}] {cell[:50]}...")
    response = client.embeddings.create(
        input=cell,
        model="text-embedding-3-large"
    )
    embedding = np.array(response.data[0].embedding)
    cell_embeddings[cell] = embedding
    print(f"        ✅ Shape: {embedding.shape}")

# Generate phenotype embeddings
print("\n[2/2] Phenotype embeddings...")
pheno_embeddings = {}
for i, pheno in enumerate(phenotypes, 1):
    print(f"  [{i}/{len(phenotypes)}] {pheno[:50]}...")
    response = client.embeddings.create(
        input=pheno,
        model="text-embedding-3-large"
    )
    embedding = np.array(response.data[0].embedding)
    pheno_embeddings[pheno] = embedding
    print(f"        ✅ Shape: {embedding.shape}")

# Save
print("\nSaving embeddings...")
np.save("../../data/embeddings/benchmark_cells.npy", cell_embeddings)
np.save("../../data/embeddings/benchmark_phenotypes.npy", pheno_embeddings)

print("\n" + "="*70)
print("✅ COMPLETE!")
print("="*70)
print(f"Cell embeddings: {len(cell_embeddings)}")
print(f"Phenotype embeddings: {len(pheno_embeddings)}")
print(f"Saved to: ../../data/embeddings/")
print("\nReady for full analysis with all 4 embeddings!")
