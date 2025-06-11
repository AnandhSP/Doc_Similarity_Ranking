import os
from agents.comparison_agent import ComparisonAgent
from agents.ranking_agent import RankingAgent
from agents.summary_agent import SummaryAgent

UPLOAD_DIR = "data/uploads"
JD_PATH = os.path.join(UPLOAD_DIR, "job_description.docx")
PROFILES_DIR = os.path.join(UPLOAD_DIR, "profiles")

def main():
    # Ensure files exist
    if not os.path.exists(JD_PATH):
        print(f"Job description not found at: {JD_PATH}")
        return

    profile_paths = [os.path.join(PROFILES_DIR, f) for f in os.listdir(PROFILES_DIR) if f.endswith(".docx")]
    if not profile_paths:
        print(f"No profile documents found in: {PROFILES_DIR}")
        return

    # Initialize agents
    comp_agent = ComparisonAgent()
    rank_agent = RankingAgent()
    summary_agent = SummaryAgent()

    # Step 1: Compute similarity
    similarity_scores = comp_agent.compute_similarity(JD_PATH, profile_paths)

    # Step 2: Rank candidates
    ranked_profiles = rank_agent.rank_profiles(similarity_scores, top_k=len(similarity_scores))

    # Step 3: Generate Summary Report
    jd_text = comp_agent.read_docx(JD_PATH)

    candidate_profiles_text = ""
    for i, path in enumerate(profile_paths):
        profile_text = comp_agent.read_docx(path)
        filename = os.path.basename(path)
        candidate_profiles_text += f"\n\nCandidate {i+1} ({filename}):\n{profile_text[:500]}..."

    ranking_results_text = ""
    for i, profile in enumerate(ranked_profiles):
        ranking_results_text += f"\n{i+1}. {profile['filename']} - Match Score: {profile['similarity_score']:.2%}"

    report = summary_agent.generate_summary(jd_text, candidate_profiles_text, ranking_results_text)

    # Step 4: Print Results
    print("\n========== RANKED PROFILES ==========")
    for rank in ranked_profiles:
        print(f"{rank['filename']} -> {rank['similarity_score']:.2%}")

    print("\n========== SUMMARY REPORT ==========")
    print(report)

if __name__ == "__main__":
    main()
