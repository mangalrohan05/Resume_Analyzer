import spacy

skill_patterns = [
    # --- HR & Management Skills ---
    {"label": "SKILL", "pattern": [{"LOWER": "fmla"}]},
    {"label": "SKILL", "pattern": [{"LOWER": "osha"}]},
    {"label": "SKILL", "pattern": [{"LOWER": "strategic"}, {"LOWER": "planning"}]},
    {"label": "SKILL", "pattern": [{"LOWER": "workers"}, {"LOWER": "compensation"}]},
    {"label": "SKILL", "pattern": [{"LOWER": "leadership"}]},
    {"label": "SKILL", "pattern": [{"LOWER": "employee"}, {"LOWER": "relations"}]},
    {"label": "SKILL", "pattern": [{"LOWER": "labor"}, {"LOWER": "union"}]},
    {"label": "SKILL", "pattern": [{"LOWER": "executive"}, {"LOWER": "management"}]},
    
    # --- Technical & Software Skills ---
    {"label": "SKILL", "pattern": [{"LOWER": "microsoft"}, {"LOWER": "excel"}]},
    {"label": "SKILL", "pattern": [{"LOWER": "excel"}]},
    {"label": "SKILL", "pattern": [{"LOWER": "access"}]},
    {"label": "SKILL", "pattern": [{"LOWER": "ms"}, {"LOWER": "word"}]},
    {"label": "SKILL", "pattern": [{"LOWER": "word"}]},
    {"label": "SKILL", "pattern": [{"LOWER": "database"}, {"LOWER": "management"}]},
    {"label": "SKILL", "pattern": [{"LOWER": "quattro"}, {"LOWER": "pro"}]},
    {"label": "SKILL", "pattern": [{"LOWER": "quark"}, {"LOWER": "express"}]},
    {"label": "SKILL", "pattern": [{"LOWER": "harvard"}, {"LOWER": "graphics"}]},

    # --- Machine Learning / AI Skills ---
    {"label": "SKILL", "pattern": [{"LOWER": "python"}]},
    {"label": "SKILL", "pattern": [{"LOWER": "machine"}, {"LOWER": "learning"}]},
    {"label": "SKILL", "pattern": [{"LOWER": "deep"}, {"LOWER": "learning"}]},
    {"label": "SKILL", "pattern": [{"LOWER": "nlp"}]},
    {"label": "SKILL", "pattern": [{"LOWER": "natural"}, {"LOWER": "language"}, {"LOWER": "processing"}]},
    {"label": "SKILL", "pattern": [{"LOWER": "spacy"}]},
    {"label": "SKILL", "pattern": [{"LOWER": "tf"}, {"LOWER": "idf"}]},
    {"label": "SKILL", "pattern": [{"LOWER": "mern"}, {"LOWER": "stack"}]},
]

class SkillExtractor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        ruler = self.nlp.add_pipe("entity_ruler", before="ner")
        ruler.add_patterns(skill_patterns)
        
    def extract(self, text: str):
        doc = self.nlp(text)
        data = {"SKILLS": set(), "ORGANIZATIONS": set(),
                "EDUCATION_CERT": set(), "EXPERIENCE_PERIODS": set()}
        for ent in doc.ents:
            if len(ent.text.strip()) < 2:
                continue
            
            if ent.label_ == "SKILL":
                data["SKILLS"].add(ent.text.title())
            elif ent.label_ in ["ORG", "NORP"]:
                data["ORGANIZATIONS"].add(ent.text)
            elif ent.label_ in ["DATE", "QUANTITY"] and any(c.isdigit() for c in ent.text):
                data["EXPERIENCE_PERIODS"].add(ent.text)
            elif any(k in ent.text for k in ["Degree", "Bachelor", "Master"]):
                data["EDUCATION_CERT"].add(ent.text)
        
        return data