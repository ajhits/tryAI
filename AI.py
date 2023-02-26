from flask import Flask, request,render_template
import nltk
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Define stopwords and stemmer
stop_words = set(nltk.corpus.stopwords.words('english'))
stemmer = nltk.stem.PorterStemmer()
lemmatizer = nltk.stem.WordNetLemmatizer()

# Create Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

# Define route for similarity comparison
@app.route('/', methods = ['GET', 'POST'])
def similarity():
    # Get the two texts from the HTML input form
    text1 = request.form['text1']
    text2 = request.form['text2']

    # Preprocess the texts
    text1_tokens = nltk.word_tokenize(text1)
    text1_tokens = [token.lower() for token in text1_tokens if token.isalpha()]
    text1_tokens = [token for token in text1_tokens if token not in stop_words]
    text1_tokens = [lemmatizer.lemmatize(token) for token in text1_tokens]
    text1_tokens = [stemmer.stem(token) for token in text1_tokens]
    text1_preprocessed = " ".join(text1_tokens)

    text2_tokens = nltk.word_tokenize(text2)
    text2_tokens = [token.lower() for token in text2_tokens if token.isalpha()]
    text2_tokens = [token for token in text2_tokens if token not in stop_words]
    text2_tokens = [lemmatizer.lemmatize(token) for token in text2_tokens]
    text2_tokens = [stemmer.stem(token) for token in text2_tokens]
    text2_preprocessed = " ".join(text2_tokens)

    # Create bag of words representation
    vectorizer = CountVectorizer().fit_transform([text1_preprocessed, text2_preprocessed])
    vectors = vectorizer.toarray()

    # Calculate cosine similarity
    similarity = cosine_similarity(vectors[0].reshape(1,-1), vectors[1].reshape(1,-1))[0][0]

    # Convert similarity to percentage level
    similarity_percentage = similarity * 100

    # Return the similarity percentage level as a response
    return f"The similarity between text1 and text2 is: {similarity_percentage}%"

if __name__ == '__main__':
    app.run(debug=True)
