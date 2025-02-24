import pandas as pd
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data files
nltk.download('stopwords')
nltk.download('wordnet')

# Load the dataset
file_path = 'C:\Users\DELL\Documents\SwiftRefund\text_emotion.csv'
df = pd.read_csv(file_path)

# Display the first few rows of the dataset
print("Dataset Preview:")
print(df.head())

# Initialize tools
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Text preprocessing function
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize the text
    tokens = text.split()
    # Remove stop words and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    # Join tokens back into a single string
    return ' '.join(tokens)

# Apply preprocessing to the 'content' column (assuming the column with text is named 'content')
df['cleaned_content'] = df['content'].apply(preprocess_text)

# Display the preprocessed data
print("\nPreprocessed Data:")
print(df[['content', 'cleaned_content']].head())

# Split data into train and test sets (assuming 'sentiment' is the target column)
X_train, X_test, y_train, y_test = train_test_split(df['cleaned_content'], df['sentiment'], test_size=0.2, random_state=42)

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Display the TF-IDF feature matrix shape
print("\nTF-IDF Feature Matrix Shape (Train):", X_train_tfidf.shape)
print("TF-IDF Feature Matrix Shape (Test):", X_test_tfidf.shape)
