# IN TERMINAL RUN
# chcp 65001

import sqlite3
import nltk
import matplotlib.pyplot as plt
from collections import OrderedDict
import random
import sys

# print(sys.stdout.encoding)

nltk.download('punkt')
nltk.download('stopwords')

conn = sqlite3.connect('reddit.db')
c = conn.cursor()


############################
def wordFilter(excluded, wordrow):
    filtered = [word for word in wordrow if word not in excluded]
    return filtered


def lowerCaseArray(wordrow):
    lowercased = [word.lower() for word in wordrow]
    return lowercased


############################

stopwords = nltk.corpus.stopwords.words('english')
# print(stopwords)


# STEM EXTRACTION

stemmer = nltk.SnowballStemmer("english")


############
def wordStemmer(wordrow):
    stemmed = [stemmer.stem(word) for word in wordrow]
    return stemmed


manual_stopwords = [',', '.', ')', '(', 'm', "'m", "n't", 'e.g', "'ve", 's', '#', '/', '``', "'s", "''", '!', 'r', ']',
                    '=', '[', 's', '&', '%', '*', '...', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '--', ';',
                    '-', ':']


############################
def data_processing(sql, manual_stopwords):
    c.execute(sql)
    row = c.fetchone()
    data = process_data_from_source(row)
    return data


def process_data_from_source(row):
    data = {'wordMatrix': [], 'all_words': []}
    interWordList = []
    interWordMatrix = []

    while row is not None:
        # wordrow = nltk.tokenize.word_tokenize(row[0] + " " + row[1])
        tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+|[^\w\s]+')
        wordrow = tokenizer.tokenize(row[0] + " " + row[1])
        wordrow_lowercased = lowerCaseArray(wordrow)
        wordrow_nostopwords = wordFilter(stopwords, wordrow_lowercased)
        wordrow_nostopwords = wordFilter(manual_stopwords, wordrow_nostopwords)
        wordrow_stemmed = wordStemmer(wordrow_nostopwords)
        interWordList.extend(wordrow_stemmed)
        interWordMatrix.append(wordrow_stemmed)
        row = c.fetchone()
        wordfreqs = nltk.FreqDist(interWordList)
        hapaxes = wordfreqs.hapaxes()
        for wordvector in interWordMatrix:
            wordvector_nohapaxes = wordFilter(hapaxes, wordvector)
            data['wordMatrix'].append(wordvector_nohapaxes)
            data['all_words'].extend(wordvector_nohapaxes)
    return data


############################

subreddits = ['bioinformatics', 'datascience']
data = {}
for subject in subreddits:
    data[subject] = data_processing(
        sql='''SELECT topicTitle,topicText,topicCategory FROM topics WHERE topicCategory = ''' + "'" + subject + "'",
        manual_stopwords=manual_stopwords)

# word_freques_cat1 = nltk.FreqDist(data['datascience']['all_words'])
# word_freques_cat2 = nltk.FreqDist(data['bioinformatics']['all_words'])
# plt.subplot(211)
# plt.hist(word_freques_cat1.values(),bins=range(10))
# plt.subplot(212)
# plt.hist(word_freques_cat2.values(),bins=range(20))
# plt.show()
# print(word_freques_cat1.hapaxes())
# print(word_freques_cat2.hapaxes())

##################################################################
holdoutLength = 50
labeled_data1 = [(word, 'datascience') for word in data['datascience']['wordMatrix'][holdoutLength:]]
labeled_data2 = [(word, 'bioinformatics') for word in data['bioinformatics']['wordMatrix'][holdoutLength:]]
labeled_data = []
labeled_data.extend(labeled_data1)
labeled_data.extend(labeled_data2)

holdout_data = data['datascience']['wordMatrix'][:holdoutLength]
holdout_data.extend(data['bioinformatics']['wordMatrix'][:holdoutLength])
holdout_data_labels = (
        [('datascience') for _ in range(holdoutLength)] + [('bioinformatics') for _ in range(holdoutLength)])

data['datascience']['all_words_dedup'] = list(OrderedDict.fromkeys(data['datascience']['all_words']))
data['bioinformatics']['all_words_dedup'] = list(OrderedDict.fromkeys(data['bioinformatics']['all_words']))

all_words = []
all_words.extend(data['datascience']['all_words_dedup'])
all_words.extend(data['bioinformatics']['all_words_dedup'])
all_words_dedup = list(OrderedDict.fromkeys(all_words))

prepared_data = [({word: (word in x[0]) for word in all_words_dedup}, x[1]) for x in labeled_data]
prepared_holdout_data = [({word: (word in x) for word in all_words_dedup}) for x in holdout_data]

random.shuffle(prepared_data)
train_size = int(len(prepared_data) * 0.75)
train = prepared_data[:train_size]
test = prepared_data[train_size:]

############# ANALYSE ###################

print("NaiveBayesClassifier:")
classifier = nltk.NaiveBayesClassifier.train(train)
print(nltk.classify.accuracy(classifier, test))
classified_data = classifier.classify_many(prepared_holdout_data)
cm = nltk.ConfusionMatrix(holdout_data_labels, classified_data)
print(cm)

# print(classifier.show_most_informative_features(20))

###### Approach of Decisions Tree  **********************
# print("DecisionTreeClassifier:")
# classifier2 = nltk.DecisionTreeClassifier.train(train)
# print(nltk.classify.accuracy(classifier2, test))
# classified_data2 = classifier2.classify_many(prepared_holdout_data)
# cm2 = nltk.ConfusionMatrix(holdout_data_labels, classified_data2)
# print(cm2)
# print(classifier2.pseudocode(depth=4))

user_test_string = "bioiformatics is cool"
data = process_data_from_source(user_test_string)
data_to_analyse = list(OrderedDict.fromkeys(data))
print(data_to_analyse)
classified_dataU = classifier.classify(data_to_analyse)
cmU = nltk.ConfusionMatrix(holdout_data_labels, classified_dataU)
print(cmU)


