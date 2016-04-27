import codecs
from random import shuffle
import nltk
from pandas import DataFrame
import os
import numpy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix, f1_score

txtpath = '/home/sunito/Desktop/spamdata/TEXT/'
respath = '/home/sunito/Desktop/out/'

def getfilenames(dir):
    return [name for name in os.listdir(dir)
            if os.path.isfile(os.path.join(dir, name))]

def readLines(file):
    with codecs.open(file, encoding="latin-1") as readtxt:
        return readtxt.read().splitlines()

def createframes(path, files, label):
    rows = []
    index = []
    #for file in files
    for file in files:
        content = "\n".join(readLines(path + file))
        rows.append({'content': content, 'class': label})
	if file in index:
            print('found duplicate: ' + file)
        else:
            index.append(file)
    data_frame = DataFrame(rows, index=index)
    return data_frame


print('load ham and spam text files')
hamtxtfiles = getfilenames(txtpath + 'ham')
spamtxtfiles = getfilenames(txtpath + 'spam')
print('ham count: ' + str(len(hamtxtfiles)))
print('spam count: ' + str(len(spamtxtfiles)) + '\n')
print('creating dataframes...')
data = DataFrame({'content': [], 'class': []})
data = data.append(createframes(txtpath + 'ham/', hamtxtfiles, 'ham'))
data = data.append(createframes(txtpath + 'spam/', spamtxtfiles, 'spam'))

print('shuffle the data... \n')
#data = data.reindex(numpy.random.permutation(data.index))
#data.to_csv(respath + 'dataframes')

print('tokenizing..using wordfrequency')
countvectorizer = CountVectorizer()
wfqs = countvectorizer.fit_transform(data['content'].values)

clf_MNB = MultinomialNB()
labels = data['class'].values
clf_MNB.fit(wfqs, labels)

print('create pipline (word frequency tokenizer) -> (MultinomialNB)')
pipeline = Pipeline([('vectorizer', CountVectorizer()), ('classifier', MultinomialNB())])

pipeline.fit(data['content'].values, data['class'].values)

print('KFold cross validation...')

k_fold= KFold(n=len(data), n_folds=6)
scores = []
confusion = numpy.array([[0,0], [0,0]])

for train_indices, test_indices in k_fold:
    train_text = data.iloc[train_indices]['content'].values
    train_y = data.iloc[train_indices]['class'].values

    test_text = data.iloc[test_indices]['content'].values
    test_y = data.iloc[test_indices]['class'].values

    pipeline.fit(train_text, train_y)
    predictions = pipeline.predict(test_text)

    confusion += confusion_matrix(test_y, predictions)
    score = f1_score(test_y, predictions, pos_label='spam')
    scores.append(score)

print('Total emails classified:', len(data))
print('Score:', sum(scores)/len(scores))
print('Confusion matrix:')
print(confusion)

