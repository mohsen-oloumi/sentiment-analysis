# -*- coding: UTF-8 -*-
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import math
from nltk.stem.wordnet import WordNetLemmatizer
import nltk
from sklearn import linear_model
tokenize = lambda doc: doc.lower().split(" ")
import gensim
import numpy as np
from gensim.models import doc2vec
from sklearn import utils
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix
LabeledSentence = gensim.models.doc2vec.LabeledSentence
from sklearn.externals import joblib

def readFile(filepath):
    word_list = []
    with open(filepath) as fp:
        for cnt, line in enumerate(fp):
            word_list.append(line)
    return word_list
def readFile_label():
    word_list = []
    filepath = './sst/label.txt'
    with open(filepath) as fp:
        for cnt, line in enumerate(fp):
            word_list.append(line)
    return word_list

def delete_stopWords(word_list):
    stop_words = set(stopwords.words('english'))
    filtered_sentence=[]
    for word_tokens in word_list:
        filtered_sentence.append([w for w in word_tokens if not w in stop_words])
    return filtered_sentence

def tokenize_word(word_list):
    tokenizer = RegexpTokenizer(r'\w+')
    word_tokens_list = []
    for word in word_list:
        word_tokens_list.append(tokenizer.tokenize(word))
    return word_tokens_list

def lemetize_words (word_list):
    i = 0
    list=[]
    for word in word_list:
        line_list = ""
        for w in word:
            tag = nltk.pos_tag([w])
            if (tag[0][1]=='VB') or (tag[0][1] == 'VBD') or (tag[0][1] == 'VBG') \
                or (tag[0][1] == 'VBN') or (tag[0][1] == 'VBP') or (tag[0][1] == 'VBZ') :
                line_list+=str((WordNetLemmatizer().lemmatize(w, "v")))+" "
                i+=1
            else:
                i+=1
                line_list+=(w)+" "
        list.append(line_list)
    return list
#############################TFIDF#######################################

def term_frequency(term, tokenized_document):
    return tokenized_document.count(term)

def sublinear_term_frequency(term, tokenized_document):
    count = tokenized_document.count(term)
    if count == 0:
        return 0
    return 1 + math.log(count)

def inverse_document_frequencies(tokenized_documents):
    idf_values = {}
    all_tokens_set = set([item for sublist in tokenized_documents for item in sublist])
    for tkn in all_tokens_set:
        contains_token = map(lambda doc: tkn in doc, tokenized_documents)
        idf_values[tkn] = 1 + math.log(len(tokenized_documents)/(sum(contains_token)))
    return idf_values

def _tfidf(documents):
    tokenized_documents = [tokenize(d) for d in documents]
    idf = inverse_document_frequencies(tokenized_documents)
    tfidf_documents = []
    for document in tokenized_documents:
        doc_tfidf = []
        for term in idf.keys():
            tf = sublinear_term_frequency(term, document)
            doc_tfidf.append(tf * idf[term])
        tfidf_documents.append(doc_tfidf)
    logesticRegiression(tfidf_documents, label_list)
    return tfidf_documents

def calTF(documents):
    tokenized_documents = [tokenize(d) for d in documents]
    idf = inverse_document_frequencies(tokenized_documents)
    tfidf_documents = []
    for document in tokenized_documents:
        doc_tfidf = []
        for term in idf.keys():
            tf = sublinear_term_frequency(term, document)
            doc_tfidf.append(tf)
        tfidf_documents.append(doc_tfidf)
    return tfidf_documents

def logesticRegiression(word_list,label_list):
    logreg = linear_model.LogisticRegression()
    logreg.fit(word_list[0:50], label_list[0:50])
    y_pred = logreg.predict(word_list[50:80])
    print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(word_list[50:80], label_list[50:80])))

###############################DOC2VEC##########################
def getVecs(model, corpus, size, vecs_type):
    vecs = np.zeros((len(corpus), size))
    for i in range(0, len(corpus)):
        index = i
        prefix = 'All_' + str(index)
        vecs[i] = model.docvecs[prefix]
    return vecs
def labelizeReviews(reviews, label_type):
    labelized = []
    for i, v in enumerate(reviews):
        label = '%s_%s' % (label_type, i)
        labelized.append(LabeledSentence(v, [label]))
    return labelized

def intitializeDoc2Vec (x_train,y_train,x_test,y_test):
    allXs = x_train + x_test
    x_train = labelizeReviews(x_train, 'Train')
    x_test = labelizeReviews(x_test, 'Test')
    allXs = labelizeReviews(allXs, 'All')
    print("B")
    model = doc2vec.Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=7)
    model.build_vocab(allXs)
    # model = doc2vec.Doc2Vec.load('doc2vec_model')
    # Pass through the data set multiple times, shuffling the training reviews each time to improve accuracy
    print("C")
    for epoch in range(20):
        model.train(utils.shuffle(x_train), total_examples=model.corpus_count, epochs=model.iter)

    train_vecs = getVecs(model, x_train, 100, 'Train')
    print train_vecs.shape

    # Construct vectors for test reviews
    test_vecs = getVecs(model, x_test, 100, 'Test')
    print test_vecs.shape
    model.save('Model_after_test')
    lr = linear_model.LogisticRegression()
    lr.fit(train_vecs, y_train)
    y_pred = lr.predict(test_vecs)
    print(y_pred)
    print 'Accuracy: %.2f' % lr.score(test_vecs, y_test)

    evaluation(y_test,y_pred)


#################LDA####################
def intitializeLDA(doc_set,label_list):
    word_list = _tfidf(doc_set)
    X_train, X_test, y_train, y_test = train_test_split(word_list, label_list, test_size=0.25)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    lda = LDA(n_components=2)
    X_train = lda.fit_transform(X_train, y_train)
    X_test = lda.transform(X_test)
    classifier = linear_model.LogisticRegression()
    classifier.fit(X_train, y_train)
    save_model(classifier,"LDA")
    print 'Accuracy: %.2f' % classifier.score(X_test, y_test)
    y_pred = classifier.predict(X_test)
    print(y_pred)

    evaluation(y_test,y_pred)
def save_model(clf,name):
    joblib.dump(clf, name+".pkl", compress=3)
#################LDA####################
def evaluation (tfidfYtest , preds):
    tp=0
    tn=0
    fp=0
    fn=0

    for i in preds:
        for j in tfidfYtest:
            if (i=='0\n' and j=='0\n'):
                tn+=1
            if (i=='1\n' and j=='1\n'):
                tp+=1
            if (i == '0\n' and j == '1\n'):
                fn += 1
            if (i == '1\n' and j == '0\n'):
                fp += 1
    precision = float(tp/float((tp+fp)))
    recall = float(tp/float((tp+fn)))
    f1 = 2*((precision*recall)/(precision+recall))
    print ("precision",precision)
    print("recall", recall)
    print ("f1" , f1)
    matrix = confusion_matrix(tfidfYtest, preds)
    print ("matrix", matrix)

if __name__ == '__main__':
    word_list = readFile('./sst/data.txt')
    label_list = readFile_label()
    word_list = tokenize_word (word_list)
    word_list = delete_stopWords (word_list)
    word_list = lemetize_words (word_list)
    #_tfidf(word_list)
    #intitializeDoc2Vec(x_train=word_list[0:50],y_train=label_list[0:50],x_test=word_list[50:80],y_test=label_list[50:80])
    intitializeLDA(word_list,label_list)

