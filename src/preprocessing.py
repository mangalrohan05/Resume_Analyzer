import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

st = set(stopwords.words('english'))
lemma = WordNetLemmatizer()

def wordnet_pos(tag):
    if tag.startswith('J'): return wordnet.ADJ
    if tag.startswith('V'): return wordnet.VERB
    if tag.startswith('N'): return wordnet.NOUN
    if tag.startswith('R'): return wordnet.ADV
    return wordnet.NOUN

def preprocessing(text : str):
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    
    words = text.split()
    pos_tags = nltk.pos_tag(words)
    
    cleaned = [lemma.lemmatize(word, wordnet_pos(tag)) for word, tag in pos_tags if word not in st]
    
    return " ".join(cleaned)