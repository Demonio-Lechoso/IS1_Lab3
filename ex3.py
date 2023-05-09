from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import re
import string
import nltk
#nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

from os.path import exists

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

#cleaning text function, takes an preferably unclean text as parameter
#Tokenization is used in NLP to split paragraphs and sentences into smaller units that can be more easily assigned meaning.
def clean(uncleanedText):
#initializing the tokenizer
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    # initializing and importing english stopwords as well as importing stemmer class
    stopWordsEng = stopwords.words('english')
    stemmer = PorterStemmer()
    # clean text and clean stemmed text lists
    cleanedText = []
    cleanedAndStemmedText = [] 

    # remove stock market notations "$AAPL", hashtags, hyperlinks,strip the "\n" characters , retweet text "RT"
    #Basically, removing special characters
    uncleanedText = re.sub(r'\$\w*', "", uncleanedText)
    uncleanedText = re.sub(r'#', '', uncleanedText)
    uncleanedText = re.sub(r'https?:\/\/.*[\r\n]*', '', uncleanedText)
    uncleanedText = uncleanedText.strip()
    uncleanedText = re.sub(r'^RT[\s]+', '', uncleanedText)

    # tokenize the treated text
    uncleanedText = tokenizer.tokenize(uncleanedText)

    # Remove punctuation and stopwords by looping the words in the uncleanedText text
    # Basically remove the stop words in the text
    for word in uncleanedText:
        if (word not in stopWordsEng and  word not in string.punctuation):
            cleanedText.append(word)

    # Stemming the words, then appending them to the list of clean stemmed text, by looping through every word in the clean text list
    for word in cleanedText:
        stemmedWord = stemmer.stem(word)  # stemming word
        cleanedAndStemmedText.append(stemmedWord)  # append to the list

    # finally return clean stemmed text
    return cleanedAndStemmedText 

#function to clean a csv text file, save the cleaned text in a new csv file and return it 
def cleanedText():
    if exists('cleanedNews.csv'): 
        return  pd.read_csv(r'cleanedNews.csv')
    else: 
        #read the news csv file
        df = pd.read_csv(r'News.csv')
        df['News']=df['News'].apply(clean)
        df['Fake']=df['Fake'].replace({True: 1, False: 0})
        #save the cleaned text in the new cleaned news csv file
        df.to_csv('cleanedNews.csv', index=False)
        return df

df = cleanedText()

x = df['News'] 
y = df['Fake']

x_train, x_test, y_train, y_test = train_test_split(x, y)

#convert text into numerical values in order to be used in the model
tf_vectorizer = TfidfVectorizer(use_idf=True)
x_train_tf = tf_vectorizer.fit_transform(x_train).toarray()
x_test_tf = tf_vectorizer.fit_transform(x_test).toarray()

# create a model
model = Sequential()

# Add an input layer
model.add(Dense(16, input_dim=x_train_tf.shape[1], activation='relu')) 

# Add first hidden layer
model.add(Dense(8, activation='relu'))

# Add second hidden layer
model.add(Dense(4, activation='relu'))

# Add output layer
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
history = model.fit(x_train_tf, y_train, epochs=50, batch_size=10)

# print the total loss and the training loss
print("Total Loss:", history.history['loss'][-1])
print("Training Loss:", history.history['val_loss'][-1])



