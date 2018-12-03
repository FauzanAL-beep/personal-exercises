
# coding: utf-8

# In[74]:


import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import os
import json
import datetime
import pandas as pd
import re


# In[75]:


df=pd.read_csv('dataset.csv',index_col=False)


# In[76]:


factory = StemmerFactory()
stemmer = factory.create_stemmer()
words = []
classes = []
documents = []
ignore_words = ['?','']
for index, pattern in df.iterrows():
    res = re.sub(r"http\S+", "", pattern['sentence'])#remove unnecessary links
    w = nltk.word_tokenize(res)
    words.extend(w)
    documents.append((w, pattern['class']))# add to documents in our corpus
    if pattern['class'] not in classes:# add to our classes list
        classes.append(pattern['class'])
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]# stem and lower each word and remove duplicates
words = list(set(words))# remove duplicates
classes = list(set(classes))# remove duplicates
print(len(documents), "documents")
print(len(classes), "classes", classes)
print(len(words), "unique stemmed words", words)


# In[77]:


training = []
output = []
output_empty = [0] * len(classes) #output array
for doc in documents: #training set, bag of words for each sentence
    bag = [] #initialize bag of words
    pattern_words = doc[0] #list of tokenized words for the pattern
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words] #stem each word
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0) #create our bag of words array
    training.append(bag)
    output_row = list(output_empty) #output is a '0' for each tag and '1' for current tag
    output_row[classes.index(doc[1])] = 1
    output.append(output_row)


# In[78]:


#Below can be replaced with SCIKIT
import numpy as np
import time

def sigmoid(x):#sigmoid function
    output = 1/(1+np.exp(-x))
    return output

def sigmoid_output_to_derivative(output):# convert output of sigmoid function to its derivative
    return output*(1-output)
 
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence) #tokenize to words
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words] #stem words
    return sentence_words

def bow(sentence, words, show_details=False): #return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
    sentence_words = clean_up_sentence(sentence) #tokenize the pattern
    bag = [0]*len(words)#bag of words
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def think(sentence, show_details=False):#this is for testing
    x = bow(sentence.lower(), words, show_details)
    if show_details:
        print ("sentence:", sentence, "\n bow:", x)
    l0 = x
    l1 = sigmoid(np.dot(l0, synapse_0)) #neural network feed forward from input to hidden
    l2 = sigmoid(np.dot(l1, synapse_1)) #neural network feed forward from hidden to output
    return l2


# In[79]:


def train(X, y, hidden_neurons=10, alpha=1, epochs=50000, dropout=False, dropout_percent=0.5):
    print ("Training with %s neurons, alpha:%s, dropout:%s %s" % (hidden_neurons, str(alpha), dropout, dropout_percent if dropout else '') )
    print ("Input matrix: %sx%s    Output matrix: %sx%s" % (len(X),len(X[0]),1, len(classes)) )
    np.random.seed(1)
    last_mean_error = 1
    #randomize weights and neurons
    synapse_0 = 2*np.random.random((len(X[0]), hidden_neurons)) - 1
    synapse_1 = 2*np.random.random((hidden_neurons, len(classes))) - 1

    prev_synapse_0_weight_update = np.zeros_like(synapse_0)
    prev_synapse_1_weight_update = np.zeros_like(synapse_1)

    synapse_0_direction_count = np.zeros_like(synapse_0)
    synapse_1_direction_count = np.zeros_like(synapse_1)
        
    for j in iter(range(epochs+1)):
        # Feed forward through layers 0, 1, and 2
        layer_0 = X
        layer_1 = sigmoid(np.dot(layer_0, synapse_0))
                
        if(dropout):
            layer_1 *= np.random.binomial([np.ones((len(X),hidden_neurons))],1-dropout_percent)[0] * (1.0/(1-dropout_percent))

        layer_2 = sigmoid(np.dot(layer_1, synapse_1))

        layer_2_error = y - layer_2

        if (j% 10000) == 0 and j > 5000:
            if np.mean(np.abs(layer_2_error)) < last_mean_error:
                print ("delta after "+str(j)+" iterations:" + str(np.mean(np.abs(layer_2_error))) )
                last_mean_error = np.mean(np.abs(layer_2_error))
            else:
                print ("break:", np.mean(np.abs(layer_2_error)), ">", last_mean_error )
                break
                
        layer_2_delta = layer_2_error * sigmoid_output_to_derivative(layer_2)

        layer_1_error = layer_2_delta.dot(synapse_1.T)

        layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)
        
        synapse_1_weight_update = (layer_1.T.dot(layer_2_delta))
        synapse_0_weight_update = (layer_0.T.dot(layer_1_delta))
        
        if(j > 0):
            synapse_0_direction_count += np.abs(((synapse_0_weight_update > 0)+0) - ((prev_synapse_0_weight_update > 0) + 0))
            synapse_1_direction_count += np.abs(((synapse_1_weight_update > 0)+0) - ((prev_synapse_1_weight_update > 0) + 0))        
        
        synapse_1 += alpha * synapse_1_weight_update
        synapse_0 += alpha * synapse_0_weight_update
        
        prev_synapse_0_weight_update = synapse_0_weight_update
        prev_synapse_1_weight_update = synapse_1_weight_update

    now = datetime.datetime.now()

    #make the model when training is done
    synapse = {'synapse0': synapse_0.tolist(), 'synapse1': synapse_1.tolist(),
               'datetime': now.strftime("%Y-%m-%d %H:%M"),
               'words': words,
               'classes': classes
              }
    synapse_file = "model.json"

    with open(synapse_file, 'w') as outfile:
        json.dump(synapse, outfile, indent=4, sort_keys=True)
    print ("saved synapses to:", synapse_file)


# In[80]:


X = np.array(training)
y = np.array(output)

start_time = time.time()

train(X, y, hidden_neurons=20, alpha=0.1, epochs=100000, dropout=False, dropout_percent=0.2) #start training

elapsed_time = time.time() - start_time
print ("processing time:", elapsed_time, "seconds")

