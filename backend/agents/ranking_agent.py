# backend/agents/ranking_agent.py

class RankingAgent:
    def rank_profiles(self, similarity_scores, top_k=3):
        sorted_profiles = sorted(similarity_scores, key=lambda x: x['similarity_score'], reverse=True)
        return sorted_profiles[:top_k]
