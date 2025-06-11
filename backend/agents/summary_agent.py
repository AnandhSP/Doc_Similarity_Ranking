from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

class SummaryAgent:
    def __init__(self):
        model_name = "google/flan-t5-large"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.summarizer = pipeline("text2text-generation", model=self.model, tokenizer=self.tokenizer)

        prompt_path = os.path.join("templates", "summary_prompt_template.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.prompt_template = f.read()

    def truncate_text(self, text, max_chars=4000):
        return text[:max_chars]

    def generate_summary(self, job_description, candidate_profiles_text, ranking_results_text):
        job_description = self.truncate_text(job_description, 1000)
        candidate_profiles_text = self.truncate_text(candidate_profiles_text, 2000)
        ranking_results_text = self.truncate_text(ranking_results_text, 1000)

        prompt = self.prompt_template.format(
            job_description=job_description,
            candidate_profiles=candidate_profiles_text,
            ranking_results=ranking_results_text
        )

        response = self.summarizer(prompt, max_length=512, do_sample=False)
        return response[0]["generated_text"]
