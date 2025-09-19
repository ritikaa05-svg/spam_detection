import pandas as pd
from nltk import accuracy
from pandas.io.pytables import performance_doc
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, classification_report
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import matplotlib
import nltk
import seaborn as sns
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

matplotlib.use('Agg')

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    ps = PorterStemmer()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

# 1. Read the CSV file with the correct encoding
file_path = r'C:\Users\Ritika Kunwar\PycharmProjects\spam\spam.csv'
try:
    df = pd.read_csv(file_path, encoding='latin-1')
except FileNotFoundError:
    print(f"Error: The file was not found at {file_path}. Please make sure the file exists and the path is correct.")
    exit()

# 2. Drop the empty columns
df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)

# 3. Rename the remaining columns for clarity
df.rename(columns={'v1': 'label', 'v2': 'message'}, inplace=True)

# 4. Check for and remove duplicate rows
df.drop_duplicates(keep='first', inplace=True)

# 5. Convert the 'label' column from 'ham' and 'spam' to numerical values (0 and 1)
encoder = LabelEncoder()
df['label'] = encoder.fit_transform(df['label'])

# 6. Apply the text transformation function
df['transformed_message'] = df['message'].apply(transform_text)

# 7. Add numerical features for analysis
df['num_characters'] = df['message'].apply(lambda x: len(x))
df['num_words'] = df['message'].apply(lambda x: len(nltk.word_tokenize(x)))
df['num_sentences'] = df['message'].apply(lambda x: len(nltk.sent_tokenize(x)))

# 8. Create and save the pie chart
plt.figure(figsize=(8, 8))
plt.pie(df['label'].value_counts(), labels=['ham', 'spam'], autopct='%1.0f%%', startangle=90,
        colors=['#ff9999', '#66b3ff'])
plt.title('Spam vs. Ham Email Distribution')
plt.savefig('spam_vs_ham_pie_chart.png')
print("Pie chart saved as 'spam_vs_ham_pie_chart.png' in your project directory.")

# 9. Create and save the histograms
# Plot 1: Histogram for num_characters
plt.figure(figsize=(12, 6))
sns.histplot(df[df['label'] == 0]['num_characters'], color='orange', label='Ham', kde=True, bins=50)
sns.histplot(df[df['label'] == 1]['num_characters'], color='red', label='Spam', kde=True, bins=50)
plt.title('Distribution of Number of Characters')
plt.legend()
plt.savefig('num_characters_histogram.png')

# Plot 2: Histogram for num_words
plt.figure(figsize=(12, 6))
sns.histplot(df[df['label'] == 0]['num_words'], color='green', label='Ham', kde=True, bins=50)
sns.histplot(df[df['label'] == 1]['num_words'], color='blue', label='Spam', kde=True, bins=50)
plt.title('Distribution of Number of Words')
plt.legend()
plt.savefig('num_words_histogram.png')

# Plot 3: Histogram for num_sentences
plt.figure(figsize=(12, 6))
sns.histplot(df[df['label'] == 0]['num_sentences'], color='purple', label='Ham', kde=True, bins=30)
sns.histplot(df[df['label'] == 1]['num_sentences'], color='brown', label='Spam', kde=True, bins=30)
plt.title('Distribution of Number of Sentences')
plt.legend()
plt.savefig('num_sentences_histogram.png')

# 10. Create the correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df[['num_characters', 'num_words', 'num_sentences']].corr(), annot=True, cmap='viridis')
plt.title('Correlation Heatmap of Message Features')
plt.savefig('correlation_heatmap.png')
print("Correlation heatmap saved as 'correlation_heatmap.png' in your project directory.")

import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Assuming your DataFrame 'df' and 'transformed_message' column are ready from previous steps.

# Create a single string of all spam messages
spam_messages = " ".join(df[df['label'] == 1]['transformed_message'])

# Generate the word cloud
spam_wc = WordCloud(width=500, height=500, min_font_size=10,
                    background_color='white').generate(spam_messages)

# Display the word cloud image
plt.figure(figsize=(10, 8))
plt.imshow(spam_wc)
plt.axis('off') # Hide the axes
plt.show()


##model Building
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
cv = CountVectorizer()
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['transformed_message']).toarray()
y = df['label'].values

# 6. Split data for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Train and evaluate the Multinomial Naive Bayes model
mnb = MultinomialNB()
mnb.fit(X_train, y_train)
y_pred1 = mnb.predict(X_test)
print("Multinomial Naive Bayes Model Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred1)}")
print("-" * 20)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred1))
print("-" * 20)
print("Classification Report:")
print(classification_report(y_test, y_pred1))

# 8. Train and evaluate the Bernoulli Naive Bayes model
bnb = BernoulliNB()
bnb.fit(X_train, y_train)  # <-- The missing line to fix the error
y_pred2 = bnb.predict(X_test)
print("\nBernoulli Naive Bayes Model Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred2)}")
print("-" * 20)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred2))
print("-" * 20)
print("Classification Report:")
print(classification_report(y_test, y_pred2))

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming you have the accuracy scores from your trained models
# Make sure to include Precision and other metrics if you want to plot them
mnb_accuracy = 0.98
bnb_accuracy = 0.97
# mnb_precision = 0.95
# bnb_precision = 0.94

# 1. Create a "wide" DataFrame with your model results
wide_df = pd.DataFrame({
    'Algorithm': ['Multinomial Naive Bayes', 'Bernoulli Naive Bayes'],
    'Accuracy': [mnb_accuracy, bnb_accuracy]
    # 'Precision': [mnb_precision, bnb_precision]
})

# 2. Melt the DataFrame into a "long" format for plotting
performance_doc = pd.melt(wide_df,
                         id_vars=['Algorithm'],
                         var_name='Metric',
                         value_name='Score')

# 3. Plot the data using seaborn.catplot
plt.figure(figsize=(10, 6))
sns.catplot(x='Algorithm', y='Score', hue='Metric', data=performance_doc, kind='bar')
plt.ylim(0.5, 1.0)
plt.title('Model Performance Comparison')
plt.show()

# 7. Define base estimators and a final estimator for stacking
estimators = [
    ('mnb', MultinomialNB()),
    ('bnb', BernoulliNB())
]
final_estimator = LogisticRegression()

# 8. Create and train the Stacking Classifier
stacked_model = StackingClassifier(estimators=estimators, final_estimator=final_estimator)
stacked_model.fit(X_train, y_train)
y_pred_stacked = stacked_model.predict(X_test)

# 9. Evaluate the stacked model
print("Stacked Model Evaluation Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_stacked)}")
print("-" * 20)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_stacked))
print("-" * 20)
print("Classification Report:")
print(classification_report(y_test, y_pred_stacked))

import pickle

# Save the TfidfVectorizer object
pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))

# Save the trained Multinomial Naive Bayes model
pickle.dump(mnb, open('model.pkl', 'wb'))










