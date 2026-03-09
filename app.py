from flask import *
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

nlp = spacy.load("en_core_web_sm")

def clean_function(text):
    text = text.lower()
    text = nlp(text)
    text = [t for t in text]
    text = [t for t in text if not t.is_stop]
    text = [t for t in text if not t.is_punct]
    text = [t.lemma_ for t in text]
    text = [str(t) for t in text]
    text = " ".join(text)
    return text

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