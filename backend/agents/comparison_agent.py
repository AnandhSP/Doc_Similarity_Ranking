# backend/agents/comparison_agent.py
import os
import docx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class ComparisonAgent:
    def __init__(self, model_name='all-MiniLM-L6-v2', chunk_size=100):
        self.model = SentenceTransformer(model_name)
        self.chunk_size = chunk_size

    def read_docx(self, filepath):
        doc = docx.Document(filepath)
        return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

    def chunk_text(self, text):
        words = text.split()
        return [" ".join(words[i:i + self.chunk_size]) for i in range(0, len(words), self.chunk_size)]

    def compute_similarity(self, jd_path, profile_paths):
        jd_text = self.read_docx(jd_path)
        jd_embedding = self.model.encode([jd_text], convert_to_tensor=True)

        results = []
        for profile_path in profile_paths:
            profile_text = self.read_docx(profile_path)
            chunks = self.chunk_text(profile_text)
            chunk_embeddings = self.model.encode(chunks, convert_to_tensor=True)
            scores = cosine_similarity(jd_embedding.cpu().numpy(), chunk_embeddings.cpu().numpy())[0]
            best_score = round(float(np.max(scores)), 4)

            results.append({
                "filename": os.path.basename(profile_path),
                "similarity_score": best_score
            })

        return results
