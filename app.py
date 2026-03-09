from flask import *
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

STOPWORDS = {
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "yourself","yourselves","he","him","his","himself","she","her","hers",
    "herself","it","its","itself","they","them","their","theirs","themselves",
    "what","which","who","whom","this","that","these","those","am","is","are",
    "was","were","be","been","being","have","has","had","having","do","does",
    "did","doing","a","an","the","and","but","if","or","because","as","until",
    "while","of","at","by","for","with","about","against","between","into",
    "through","during","before","after","above","below","to","from","up","down",
    "in","out","on","off","over","under","again","further","then","once","here",
    "there","when","where","why","how","all","both","each","few","more","most",
    "other","some","such","no","nor","not","only","own","same","so","than","too",
    "very","can","will","just","should","now","would","could","also"
}

def clean_function(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)

app = Flask(__name__)

def extract_text_pdf(pdf):
    reader = PdfReader(pdf)
    text = ""
    for page in reader.pages:
        text = text + page.extract_text()
    return text


@app.route("/", methods=["GET","POST"])
def home():
    if request.method == "POST":
        resume_pdf = request.files.get("resume_pdf")
        jd_pdf = request.files.get("jd_pdf")

        resume_text = extract_text_pdf(resume_pdf)
        jd_text = extract_text_pdf(jd_pdf)
        
        cleaned_resume = clean_function(resume_text)
        cleaned_jd = clean_function(jd_text)

        tv = TfidfVectorizer()
        vectors = tv.fit_transform([cleaned_resume, cleaned_jd])
        score = cosine_similarity(vectors[0], vectors[1])
        num = score.flatten()[0]
        msg ="Your score is " +str(round(num,2)*100) + "%"
        return render_template("home.html", msg=msg)

    else:
        return render_template("home.html")

if __name__ == "__main__":
    app.run(debug=True, use_reloader=True)