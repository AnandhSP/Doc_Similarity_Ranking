# backend/agents/summarize_agent.py
import os
import docx
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from datetime import datetime
import re

class SummarizeAgent:
    def __init__(self, model_name='microsoft/DialoGPT-medium'):
    
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize the text generation pipeline
        try:
            self.summarizer = pipeline(
                "text-generation",
                model=model_name,
                tokenizer=model_name,
                device=0 if torch.cuda.is_available() else -1,
                max_length=512,
                truncation=True
            )
        except:
            # Fallback to a lighter model if the primary one fails
            self.summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=0 if torch.cuda.is_available() else -1,
                max_length=150,
                min_length=50,
                truncation=True
            )
    
    def read_docx(self, filepath):
        """Read content from a DOCX file"""
        doc = docx.Document(filepath)
        return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    
    def extract_key_requirements(self, jd_text):
        """Extract key requirements from job description using AI"""
        prompt = f"""
        Analyze this job description and extract key requirements in categories:
        Job Description: {jd_text[:1000]}...
        
        Extract:
        1. Technical Skills
        2. Experience Level
        3. Key Responsibilities
        4. Preferred Qualifications
        """
        
        try:
            # Use AI to analyze JD
            analysis = self.summarizer(prompt, max_length=200, num_return_sequences=1)
            if isinstance(analysis, list) and len(analysis) > 0:
                extracted_text = analysis[0].get('generated_text', '') or analysis[0].get('summary_text', '')
            else:
                extracted_text = str(analysis)
        except:
            # Fallback to basic extraction
            extracted_text = self._basic_requirement_extraction(jd_text)
        
        return self._parse_requirements(extracted_text, jd_text)
    
    def _basic_requirement_extraction(self, jd_text):
        """Fallback method for requirement extraction"""
        # Basic keyword extraction
        tech_keywords = ['python', 'java', 'react', 'node', 'sql', 'aws', 'docker', 'kubernetes']
        exp_keywords = ['years', 'experience', 'senior', 'junior', 'lead']
        
        found_tech = [word for word in tech_keywords if word.lower() in jd_text.lower()]
        found_exp = [phrase for phrase in jd_text.split('.') if any(keyword in phrase.lower() for keyword in exp_keywords)]
        
        return f"Technical Skills: {', '.join(found_tech)}\nExperience: {found_exp[0] if found_exp else 'Not specified'}"
    
    def _parse_requirements(self, extracted_text, original_jd):
        """Parse extracted requirements into structured format"""
        requirements = {
            'technical_skills': [],
            'experience_level': '',
            'key_responsibilities': [],
            'preferred_qualifications': []
        }
        
        # Extract technical skills using regex and keywords
        tech_pattern = r'(?:python|java|javascript|react|angular|vue|node|sql|postgresql|mysql|aws|azure|gcp|docker|kubernetes|git|api|rest|microservices|django|flask|spring)'
        tech_skills = list(set(re.findall(tech_pattern, original_jd.lower())))
        requirements['technical_skills'] = tech_skills
        
        # Extract experience level
        exp_pattern = r'(\d+)[\+\-\s]*(?:years?|yrs?)'
        exp_match = re.search(exp_pattern, original_jd.lower())
        if exp_match:
            requirements['experience_level'] = f"{exp_match.group(1)}+ years"
        
        return requirements
    
    def analyze_candidate_profile(self, profile_text, requirements):
        """Analyze individual candidate profile against requirements"""
        prompt = f"""
        Analyze this candidate profile against job requirements:
        
        Requirements: {str(requirements)}
        Profile: {profile_text[:800]}...
        
        Rate the candidate on:
        1. Technical Skills Match (1-5 stars)
        2. Experience Level Match (1-5 stars)  
        3. Key Strengths
        4. Potential Gaps
        """
        
        try:
            analysis = self.summarizer(prompt, max_length=150, num_return_sequences=1)
            if isinstance(analysis, list) and len(analysis) > 0:
                ai_analysis = analysis[0].get('generated_text', '') or analysis[0].get('summary_text', '')
            else:
                ai_analysis = str(analysis)
        except:
            ai_analysis = self._basic_profile_analysis(profile_text, requirements)
        
        return self._parse_candidate_analysis(profile_text, requirements, ai_analysis)
    
    def _basic_profile_analysis(self, profile_text, requirements):
        """Fallback method for profile analysis"""
        profile_lower = profile_text.lower()
        
        # Count matching technical skills
        tech_matches = [skill for skill in requirements['technical_skills'] if skill in profile_lower]
        tech_score = min(5, len(tech_matches))
        
        # Check experience level
        exp_pattern = r'(\d+)[\+\-\s]*(?:years?|yrs?)'
        exp_matches = re.findall(exp_pattern, profile_lower)
        exp_score = 3  # Default
        
        if exp_matches:
            max_exp = max([int(match) for match in exp_matches])
            required_exp = int(requirements['experience_level'].split('+')[0]) if requirements['experience_level'] else 3
            exp_score = min(5, max(1, (max_exp / required_exp) * 3))
        
        return f"Technical Skills: {tech_score}/5 stars\nExperience: {exp_score}/5 stars\nStrengths: {', '.join(tech_matches)}"
    
    def _parse_candidate_analysis(self, profile_text, requirements, ai_analysis):
        """Parse AI analysis into structured format"""
        profile_lower = profile_text.lower()
        
        # Extract technical skills match
        tech_matches = [skill for skill in requirements['technical_skills'] if skill in profile_lower]
        tech_score = min(5, len(tech_matches) + 1)
        
        # Extract experience
        exp_pattern = r'(\d+)[\+\-\s]*(?:years?|yrs?)'
        exp_matches = re.findall(exp_pattern, profile_lower)
        exp_years = max([int(match) for match in exp_matches]) if exp_matches else 0
        
        # Leadership indicators
        leadership_keywords = ['lead', 'manage', 'team', 'mentor', 'senior', 'principal']
        leadership_score = sum(1 for keyword in leadership_keywords if keyword in profile_lower)
        
        return {
            'technical_score': tech_score,
            'experience_years': exp_years,
            'leadership_score': min(5, leadership_score),
            'strengths': tech_matches,
            'ai_insights': ai_analysis[:200] if ai_analysis else "Standard analysis completed"
        }
    
    def generate_comparative_summary(self, ranked_results, jd_path, profile_paths):
        """Generate the main comparative summary report"""
        # Read job description
        jd_text = self.read_docx(jd_path)
        position_title = self._extract_position_title(jd_text)
        
        # Extract requirements
        requirements = self.extract_key_requirements(jd_text)
        
        # Analyze each candidate
        candidate_analyses = []
        for result in ranked_results:
            # Find the full path for this candidate
            candidate_path = None
            for path in profile_paths:
                if os.path.basename(path) == result['filename']:
                    candidate_path = path
                    break
            
            if candidate_path:
                profile_text = self.read_docx(candidate_path)
                analysis = self.analyze_candidate_profile(profile_text, requirements)
                candidate_analyses.append({
                    'filename': result['filename'],
                    'similarity_score': result['similarity_score'],
                    'analysis': analysis
                })
        
        # Generate the comparative report
        return self._format_comparative_report(position_title, candidate_analyses, requirements)
    
    def _extract_position_title(self, jd_text):
        """Extract position title from job description"""
        lines = jd_text.split('\n')
        for line in lines[:5]:  # Check first 5 lines
            if any(keyword in line.lower() for keyword in ['position', 'role', 'job title', 'opening']):
                return line.strip()
        return "Position Not Specified"
    
    def _format_comparative_report(self, position_title, candidate_analyses, requirements):
        """Format the final comparative summary report"""
        report = []
        report.append("=== CANDIDATE COMPARISON MATRIX ===")
        report.append(f"Position: {position_title}")
        report.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Side-by-side comparison table
        report.append("--- SIDE-BY-SIDE COMPARISON ---")
        report.append("")
        
        if len(candidate_analyses) >= 3:
            cand1, cand2, cand3 = candidate_analyses[:3]
            
            # Create comparison table
            report.append("| CRITERIA           | {} ({:.1f}%) | {} ({:.1f}%) | {} ({:.1f}%) |".format(
                cand1['filename'].replace('.docx', ''), cand1['similarity_score']*100,
                cand2['filename'].replace('.docx', ''), cand2['similarity_score']*100,
                cand3['filename'].replace('.docx', ''), cand3['similarity_score']*100
            ))
            report.append("|" + "-"*19 + "|" + "-"*20 + "|" + "-"*23 + "|" + "-"*19 + "|")
            
            # Technical Skills Row
            report.append("| Technical Skills   | {} | {} | {} |".format(
                "⭐" * cand1['analysis']['technical_score'],
                "⭐" * cand2['analysis']['technical_score'],
                "⭐" * cand3['analysis']['technical_score']
            ))
            
            # Experience Row
            report.append("| Experience         | {} years | {} years | {} years |".format(
                cand1['analysis']['experience_years'],
                cand2['analysis']['experience_years'],
                cand3['analysis']['experience_years']
            ))
            
            # Leadership Row
            report.append("| Leadership         | {} | {} | {} |".format(
                "⭐" * cand1['analysis']['leadership_score'],
                "⭐" * cand2['analysis']['leadership_score'],
                "⭐" * cand3['analysis']['leadership_score']
            ))
        
        report.append("")
        report.append("--- KEY DIFFERENTIATORS ---")
        report.append("")
        
        # Individual candidate highlights
        for i, candidate in enumerate(candidate_analyses[:3]):
            rank_emoji = ["🥇", "🥈", "🥉"][i]
            name = candidate['filename'].replace('.docx', '').replace('_', ' ').title()
            
            report.append(f"{rank_emoji} {name} - {candidate['similarity_score']*100:.1f}% Match")
            report.append(f"   • Strengths: {', '.join(candidate['analysis']['strengths'][:3])}")
            report.append(f"   • Experience: {candidate['analysis']['experience_years']} years")
            report.append(f"   • AI Insight: {candidate['analysis']['ai_insights'][:100]}...")
            report.append("")
        
        report.append("--- QUICK DECISION MATRIX ---")
        if candidate_analyses:
            best_candidate = candidate_analyses[0]['filename'].replace('.docx', '').replace('_', ' ').title()
            report.append(f"Best overall match: {best_candidate}")
            report.append(f"Recommended for interview: Top 2 candidates")
            report.append(f"Technical requirements coverage: {len(requirements['technical_skills'])} key skills identified")
        
        return "\n".join(report)
