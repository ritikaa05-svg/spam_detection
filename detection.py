import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import matplotlib
import nltk
import seaborn as sns

# Set the backend to 'Agg' to avoid display errors
matplotlib.use('Agg')

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

# 6. Calculate the number of characters, words, and sentences
# Make sure you have downloaded the 'punkt' and 'punkt_tab' resources from NLTK.
# Run 'nltk.download('punkt')' and 'nltk.download('punkt_tab')' in your terminal if you haven't already.
df['num_characters'] = df['message'].apply(lambda x: len(x))
df['num_words'] = df['message'].apply(lambda x: len(nltk.word_tokenize(x)))
df['num_sentences'] = df['message'].apply(lambda x: len(nltk.sent_tokenize(x)))

# 7. Create and save the pie chart
plt.figure(figsize=(8, 8))
plt.pie(df['label'].value_counts(), labels=['ham', 'spam'], autopct='%1.0f%%', startangle=90, colors=['#ff9999','#66b3ff'])
plt.title('Spam vs. Ham Email Distribution')
plt.savefig('spam_vs_ham_pie_chart.png')
print("Pie chart saved as 'spam_vs_ham_pie_chart.png' in your project directory.")

# 8. Create and save the histograms
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

plt.figure(figsize=(10, 8))
sns.heatmap(df[['num_characters', 'num_words', 'num_sentences']].corr(), annot=True, cmap='viridis')
plt.title('Correlation Heatmap of Message Features')
plt.savefig('correlation_heatmap.png')

print("Correlation heatmap saved as 'correlation_heatmap.png' in your project directory.")

##Data preprocessing
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    return text
transform_text('I am Ritika Kunwar')