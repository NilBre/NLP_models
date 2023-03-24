# -*- coding: utf-8 -*-
import os
import sys
import keras
from keras import layers
import tensorflow.keras
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, LSTM
from keras.layers.embeddings import Embedding
from keras.layers import Input
from keras.layers import SpatialDropout1D
from keras.layers import BatchNormalization
from keras.layers.merge import Concatenate
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.constraints import maxnorm
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import SGD
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, Callback, LearningRateScheduler
from tensorflow.keras import regularizers
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import multilabel_confusion_matrix, classification_report, confusion_matrix

import numpy as np
import pandas as pd
import string
import matplotlib.pyplot as plt
from tqdm import tqdm
import re
from collections import Counter
from itertools import chain

from label_extractor import full_DataFrames
from label_extractor import get_label_names

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import scipy.sparse as sp_sparse
from sklearn.preprocessing import MultiLabelBinarizer
import warnings

warnings.simplefilter("ignore")
warnings.warn("deprecated", DeprecationWarning)
warnings.simplefilter("ignore")

german_stop_words = stopwords.words('german')

def RemoveExamples(labels,examples,thres=.1):
    thresh = .1
    label_hist,bins = np.histogram(labels,bins=13,density=True)
    to_remove = []
    for i in range(len(labels)):
        if label_hist[int(labels[i])] <= thresh:
            to_remove.append(i)
    labels=np.delete(labels,to_remove,0)
    examples=np.delete(examples,to_remove,0)
    return labels, examples


def HandlingIsCorrect(labels,duplicates):
    correct = False
    if (len(labels) - np.sum(duplicates) - len(duplicates)) == 0:
        correct = True
    return correct


def HandleMultLabels(labels):
    list_of_duplicates = [] ### list of duplicates, we have to create from every example
    list_of_labels = []
    nlines, ncols = labels.shape
    for i in range(nlines):
        nlabels = 0
        for k in range(ncols):
            if labels[i][k] == 1:
                list_of_labels.append(k)
                nlabels += 1
        list_of_duplicates.append(nlabels-1)
    if HandlingIsCorrect(list_of_labels,list_of_duplicates) is True:
        return list_of_labels, np.asarray(list_of_duplicates)
    else:
       raise ValueError('Multiple Labels not handled correctly. This is bad ... ')


def DuplicateExamples(df,list_of_duplicates):
    duplicated_df = np.repeat(df,list_of_duplicates+1,axis=0)
    return duplicated_df


# diese methode muss angepasst werden sobald die umlaute alle stimmen
def prep(sen):
    sen = re.sub(r"\s+[a-zA-Z]\s+", ' ', sen)
    sen = re.sub('[0-9]', '', sen)
    sen = sen.lower()
    sen = sen.strip(' ')
    return sen

# wirft alle links aus den texten
def strip_links(text):
    link_regex    = re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', re.DOTALL)
    links         = re.findall(link_regex, text)
    for link in links:
        text = text.replace(link[0], ', ')
    return text

# wirft alle hashtags und personenmarkierungen ueber '@' heraus
def strip_all_entities(text):
    entity_prefixes = ['@','#']
    for separator in  string.punctuation:
        if separator not in entity_prefixes :
            text = text.replace(separator,'')
    words = []
    for word in text.split():
        word = word.strip()
        if word:
            if word[0] not in entity_prefixes:
                words.append(word)
    return ' '.join(words)

# wortzähler
def count_words(text):
    count = Counter()
    for i in text:
        for word in i.split():
            count[word] += 1
    return count

# nur zum plotten der loss parameter und accuracies
plt.style.use('ggplot')
def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig('/home/nbreer/Pictures/GYF_pictures/NN_acc_loss_fold_{}.pdf'.format(fold_no))

# y = labels saved from label_extractor.py
labels = pd.read_csv('labels.csv', delimiter=',')
# X = texte generiert mit GenCorrectedTexts.py
X_newtext = pd.read_csv('texte_corrected.csv')
# print(labels)
# print(X_newtext)
# speichere in X nun die texte welche durch obige preparationen laufen
X = []
sentences = list(X_newtext["text"])
for sen in sentences:
    X.append(strip_all_entities(strip_links(prep(sen))))

N = count_words(X)
X_tokenize = []
numword = 0
for sen in X:
    text_tokens = word_tokenize(sen)
    tokens_without_sw = [word for word in text_tokens if not word in german_stop_words]
    filtered_sen = (' ').join(tokens_without_sw)
    X_tokenize.append(filtered_sen)
    numword += 1


# num_word ist die anzahl an uniquen worten
num_word = len(N)
#
# lab_array = labels.to_numpy()
# labels, duplicates = HandleMultLabels(lab_array)
# examples = DuplicateExamples(X_tokenize,duplicates)
# labs, examples = RemoveExamples(labels,examples,thres=.1)

# split in train and test
# X_train, X_test, y_train, y_test = train_test_split(examples, labs, test_size=0.33, random_state=42, shuffle=True)
X_train, X_test, y_train, y_test = train_test_split(X_tokenize, labels, test_size=0.33, random_state=42, shuffle=True)
# make tokenization to binarize sentences-> speichere worte als zahlen aus dict
tokenizer = Tokenizer(num_words=num_word)
tokenizer.fit_on_texts(X_train)

# save texts as number sequences -> wort = zahlenkette
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

# main settings
N_vocabulary = len(tokenizer.word_index) + 1 # vocab size
sen_length = 200  # freier parameter: maximalanzahl an worten pro tweet
epochs = 100  # anzahl an lern-epochen
embedding_dim = 300  # glove file dimension
num_folds = 10  # anzahl an folds fuer cross validation
batch_size = 20  # batchsize

# pad the texts so they have all same length->
# if text is shorter than sen_length, fill the rest with zeros
X_train = pad_sequences(X_train, padding='post', maxlen=sen_length)
X_test = pad_sequences(X_test, padding='post', maxlen=sen_length)

# create embedding matrix with Glove file
def create_embedding_matrix(filepath, word_index, embedding_dim):
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    with open(filepath) as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word]
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix

# pre embedded glove file with dimension 300
Glove_file = '/home/nbreer/glove_files/vectors_300d.txt'  # deutsches glove file von: https://deepset.ai/german-word-embeddings
embedding_matrix = create_embedding_matrix(Glove_file, tokenizer.word_index, embedding_dim)
# die übereinstimmung zwischen embedding matrix worten und den von uns verwendeten in den tweets
nonzero_elements = np.count_nonzero(np.count_nonzero(embedding_matrix, axis=1))
print(nonzero_elements / N_vocabulary)

acc_per_fold = []
loss_per_fold = []

# inputs = X, targets = y
inputs = np.concatenate((X_train, X_test), axis=0)
targets = np.concatenate((y_train, y_test), axis=0)
# Define the K-fold Cross Validator
kfold = KFold(n_splits=num_folds, shuffle=True)

fold_no = 1

# speichere die loss und acc werte weg und plotte spaeter
ACC = []
VAL_ACC = []
LOSS = []
VAL_LOSS = []

# output params from classification report
micro_precision = []
micro_recall = []
micro_f1 = []

macro_precision = []
macro_recall = []
macro_f1 = []

accurcay = []

for train, test in kfold.split(inputs, targets):
    # model definition
    model = Sequential()  # für sequential daten, z.b. texte
    model.add(Embedding(N_vocabulary, embedding_dim, weights=[embedding_matrix], input_length=sen_length, trainable=True))  # eine embedding layer
    # outputdim from embedding is (10,100,200) -> (batchsize, embedding_dim, sen_length)
    model.add(layers.Conv1D(128, 3, activation='relu'))
    # model.add(layers.Conv1D(64, 3, activation='relu'))
    model.add(Dropout(0.5))  # overtraining reduzieren
    model.add(layers.GlobalMaxPooling1D())  # reduce diemnsionality and also overtraining
    model.add(BatchNormalization())  # skaliert den shift in den hidden layer values

    # model.add(Dense(64, activation='relu'))
    # model.add(Dropout(0.25))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(13, activation='sigmoid'))

    # model.summary()
    # sgd = SGD(lr=0.05, decay=1e-1, momentum=0.95)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

    # callbacks
    # reduziert die lern rate wenn der validation loss auf ein plateau stößt
    lrPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.001, patience=5, min_lr=1e-6)
    # wenn der validation loss sich stark verschlechtert oder konstant bleibt (für 15 epochen) vorher stoppen
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=12, mode='auto')
    # model fitten
    history = model.fit(inputs[train], targets[train], batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(inputs[test], targets[test]), callbacks=[lrPlat, early_stop])
    # predicten
    score = model.evaluate(inputs[test], targets[test], verbose=1)

    predictions = model.predict(inputs[test])
    predictions = (predictions > 0.5).astype('int32')
    y_true = targets[test]
    y_pred = predictions
    # con_mat = multilabel_confusion_matrix(y_true, y_pred)

    # print(con_mat)
    report = classification_report(y_true, y_pred, output_dict=True)
    # print(report)
    # fill values and analyze later
    micro_precision.append(report['micro avg']['precision'])
    micro_recall.append(report['micro avg']['recall'])
    micro_f1.append(report['micro avg']['f1-score'])
    macro_precision.append(report['macro avg']['precision'])
    macro_recall.append(report['macro avg']['recall'])
    macro_f1.append(report['macro avg']['f1-score'])

    print('####################################')
    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {score[0]}; {model.metrics_names[1]} of {score[1]*100}%')
    acc_per_fold.append(score[1] * 100)
    loss_per_fold.append(score[0])

    plot_history(history)

    acc = history.history['acc']
    ACC.append(acc)
    val_acc = history.history['val_acc']
    VAL_ACC.append(val_acc)
    loss = history.history['loss']
    LOSS.append(loss)
    val_loss = history.history['val_loss']
    VAL_LOSS.append(val_loss)

    fold_no = fold_no + 1
print('#############################')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')

plt.clf()

plt.figure(figsize=(12, 12))
for count in range(num_folds):
    xacc = range(1, len(ACC[count]) + 1)
    plt.subplot(2, 2, 1)
    plt.plot(xacc, ACC[count], label='Training acc, fold#{}'.format(count))
    plt.title('Training accuracy')
    plt.legend()
    xvalacc = range(1, len(VAL_ACC[count]) + 1)
    plt.subplot(2, 2, 2)
    plt.plot(xvalacc, VAL_ACC[count], label='Validation acc, fold#{}'.format(count))
    plt.title('Validation accuracy')
    plt.legend()
    xloss = range(1, len(LOSS[count]) + 1)
    plt.subplot(2, 2, 3)
    plt.plot(xloss, LOSS[count], label='Training loss, fold#{}'.format(count))
    plt.title('Training loss')
    plt.legend()
    xvaloss = range(1, len(VAL_LOSS[count]) + 1)
    plt.subplot(2, 2, 4)
    plt.plot(xvaloss, VAL_LOSS[count], label='Validation loss, fold#{}'.format(count))
    plt.title('Validation loss')
    plt.legend()
plt.legend(loc='best')
plt.savefig('/home/nbreer/Pictures/GYF_pictures/accuracy_loss.pdf')
plt.clf()
# outputs from report
mac_mean_r = np.mean(macro_recall)
mac_mean_p = np.mean(macro_precision)
mac_mean_f1 = np.mean(macro_f1)
mic_mean_r = np.mean(micro_recall)
mic_mean_p = np.mean(micro_precision)
mic_mean_f1 = np.mean(micro_f1)
print('##############################################')
print('Mittelwerte der classification report scores: ')
print('##############################################')
print('mean macro recall_score: ', mac_mean_r)
print('mean macro precision_score: ', mac_mean_p)
print('mean macro f1-score: ', mac_mean_f1)
print('mean micro recall_score: ', mic_mean_r)
print('mean micro precision_score: ', mic_mean_p)
print('mean micro f1-score: ', mic_mean_f1)
print('##############################################')

x = np.linspace(0, num_folds-1, num_folds)
plt.plot(x, macro_precision, 'r--', label='macro_precision')
plt.plot(x, macro_recall, 'g--', label='macro_recall')
plt.plot(x, macro_f1, 'b--', label='macro_f1')
plt.legend()
plt.xlabel('fold number')
plt.ylabel('score')
plt.title('macro scores')
plt.savefig('/home/nbreer/Pictures/GYF_pictures/macro_scores.pdf')
plt.clf()

plt.plot(x, micro_precision, 'r--', label='micro_precision')
plt.plot(x, micro_recall, 'g--', label='micro_recall')
plt.plot(x, micro_f1, 'b--', label='micro_f1')
plt.legend()
plt.xlabel('label number')
plt.ylabel('score')
plt.title('micro scores')
plt.savefig('/home/nbreer/Pictures/GYF_pictures/micro_scores.pdf')
'''
to do: - ist es sinnvoll die confusions matrizen als 2x2 für jeden
         fold wegzuspeichern oder lieber sein lassen?
       - den classification report finde ich sinnvoll (sinn voll wegspeicher j/n ?)
       - precision recall kurve plotten
       - die doppelten datensaetze rausnehmen + die beiden ausstehenden einbinden
       - stopwords rausnehmen
#############################
Average scores for all folds:
> Accuracy: 41.99999988079071 (+- 7.628073621980891)
> Loss: 0.3152141809463501
##############################################
Mittelwerte der classification report scores:
##############################################
mean macro recall_score:  0.08531814783981387
mean macro precision_score:  0.1245824914801168
mean macro f1-score:  0.08762457651165374
mean micro recall_score:  0.24985597858102998
mean micro precision_score:  0.47118148342341365
mean micro f1-score:  0.32564299143554326
##############################################
'''
