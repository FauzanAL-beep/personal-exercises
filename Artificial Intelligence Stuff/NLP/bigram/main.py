
# coding: utf-8

# In[98]:


import nltk
import math
#from nltk.util import ngrams
from collections import Counter
#nltk.download('punkt')
def find_ngrams(input_list, n):
    return zip(*[input_list[i:] for i in range(n)])
#sdef perplex():
    
with open('wikipedia.txt', 'r', encoding='utf-8') as myfile:
    data=myfile.read().replace('\n', '')
    
words = nltk.word_tokenize(data)
#print(words)

bigrams = find_ngrams(words, 2)
#bigrams = ngrams(words,2)
#print(list(bigram))


# model = nltk.FreqDist(bigrams)
# for k,v in model.items():
#     if(k[0]=="adalah"):
#         print(k,v)
    
model = Counter(bigrams)
print(model.most_common(100))


#my_bigrams = nltk.bigrams(words)
#print(type( my_bigrams))
#my_trigrams = nltk.trigrams(words)


# In[74]:


#print(type(bigrams))
#print(model.keys() )
#model['adalah']
already = 0
wword = input('predict the next word after word: ')
predicted_word = ''
pword_counter = 0
for k,v in model.most_common():
    if(k[0]==wword):
        pword_counter += 1
for k,v in model.most_common():
    if(k[0]==wword):
        if(already==0):
            already=1
            predicted_word=k[1]
        #print(k,"with probability:",v/pword_counter)
print('')
print('predicted word after word "'+wword+'" is "'+predicted_word+'"')


# In[113]:


sentence = input('Enter a sentence to be analyzed: ')
sentence_map = nltk.word_tokenize(sentence)
print(sentence_map)
#print(len(sentence_map))
 #first word counter
i=0
prob=1
while i<len(sentence_map)-1:
    print(sentence_map[i],sentence_map[i+1])
    fwordc=0
    for k,v in model.most_common():
        if(k[0]==sentence_map[i]):
            fwordc += 1
    for k,v in model.most_common():
        if(k[0]==sentence_map[i] and k[1]==sentence_map[i+1]):
            prob *= v/fwordc
            print(k,v)
    i+=1
print(prob)
perplexity = (1/prob)**(1/len(sentence_map))
print('perplexity is: ',perplexity)

