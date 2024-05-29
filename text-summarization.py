import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import string

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text: str) -> list:
    """Preprocess text by tokenizing and removing stopwords and punctuation."""
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    words = [word.lower() for word in words if word.lower() not in stop_words and word not in string.punctuation]
    return words

def score_sentences(sentences: list, word_frequencies: dict) -> dict:
    """Score sentences based on word frequencies."""
    sentence_scores = {}
    for sentence in sentences:
        word_count_in_sentence = len(word_tokenize(sentence))
        for word in word_tokenize(sentence.lower()):
            if word in word_frequencies:
                if sentence not in sentence_scores:
                    sentence_scores[sentence] = word_frequencies[word]
                else:
                    sentence_scores[sentence] += word_frequencies[word]
        sentence_scores[sentence] = sentence_scores[sentence] / word_count_in_sentence
    return sentence_scores

def generate_summary(text: str, num_sentences: int = 3) -> str:
    """Generate a summary of the text by scoring and ranking sentences."""
    sentences = sent_tokenize(text)
    words = preprocess_text(text)
    
    # Calculate word frequencies
    word_frequencies = {}
    for word in words:
        word_frequencies[word] = word_frequencies.get(word, 0) + 1
    
    # Score sentences
    sentence_scores = score_sentences(sentences, word_frequencies)
    
    # Select top sentences
    ranked_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)
    summary = ''.join(ranked_sentences[:num_sentences])
    
    return summary

if __name__ == "__main__":
    text = input("Enter something to summarize: ")
    summary = generate_summary(text)
    print("Summarized:")
    print(summary)
