
# coding: utf-8

# In[ ]:


import nltk
nltk.download('punkt')
import math
from collections import Counter

def ngrams(input_list, n):
    return zip(*[input_list[i:] for i in range(n)]) #convert tokens to a bigram model

with open('wiki100_1.txt', 'r', encoding='utf-8') as myfile: #loading text file (corpus) to string data
    data=myfile.read().replace('\n', '')
    
words = nltk.word_tokenize(data) #tokenize texts to words as a list element

bigrams = ngrams(words, 2) #convert tokens to a bigram model

model = Counter(bigrams) #add frequency of occurence to each bigram
#print(model)#.most_common(100))


# In[ ]:


already = 0
wword = input('predict the next word after word: ') #input the desired word to be predicted
predicted_word = ''
pword_counter = 0
probs=0
for k,v in model.most_common():
    if(k[0]==wword):
        pword_counter += v #counting the total occurence of the desired word
for k,v in model.most_common():
    if(k[0]==wword): #choosing the first element since the list is sorted descendingly, the higher in the list has bigger probability
        if(already==0):
            already=1
            predicted_word=k[1] #assign the word after the desired word that was inputted
            probs=v
print('')
print('predicted word after word "'+wword+'" is "'+predicted_word+'" with probability: ',probs,'/',pword_counter) #output


# In[ ]:


print('PERPLEXITY WITH SMOOTHING')
sentence = input('Enter a sentence to be analyzed: ') #input a sentence to be checked its perplexity
sentence_map = nltk.word_tokenize(sentence) #conver the sentence to list of words
print(sentence_map)
voc = len(Counter(sentence_map)) #counting the number of words uniquely
print('')
print('|V|=',voc)
prob=1
i=0
while i<len(sentence_map)-1: #iterate through list of words
    print(sentence_map[i],sentence_map[i+1])
    adaf=False #first word boolean
    adas=False #sedcond word boolean
    notbigram=True #in bigram boolean
    fwordc=0
    for k,v in model.most_common(): #iterate to check if first or second word exists
        if(k[0]==sentence_map[i]):
            fwordc += v #counting the occurence of first word
        if(k[0]!=sentence_map[i] and k[1]==sentence_map[i+1]):
            adas=True #if second word exists
        elif(k[0]==sentence_map[i] and k[1]!=sentence_map[i+1]):
            adaf=True #if first word exists
    if(adaf==True and adas==True):
        for k,v in model.most_common(): #iterate through bigram to check if this bigram exists           
            if(k[0]==sentence_map[i] and k[1]==sentence_map[i+1]): #if bigram occurence exists
                prob *= v+1/(fwordc+voc) #update/count probability
                print(k,v)
                print('probability before smoothing: ',v,'/',fwordc)
                print('probability after smoothing: ',v+1,'/',fwordc+voc)
                notbigram=False #assign the fact that both words exist and did occur in bigram
                break
        if(notbigram==True):
            prob *= 1/(fwordc+voc) #since it's with smoothing technique, the unknown occurence is taken into account to be counted
            print('probability before smoothing: not counted')
            print('probability after smoothing:',1,'/',fwordc+voc)
        
    if(adaf==False and adas==True):
        print("second word exist in the bigram but the first word doesn't") #this happens if the first word does never exist in bigram model
    elif(adaf==True and adas==False):
        print("first word exist in the bigram but the second word doesn't") #this happens if the second word does never exist in bigram model
    elif(adaf==False and adas==False):
        print("Both words don't exist in the bigram")  #this happens if both words never exist in bigram model
    print('')
    i+=1
print(prob) #counting the probability of the sentence
perplexity = (1/prob)**(1/len(sentence_map)) #counting the perplexity
print('perplexity is: ',perplexity)

