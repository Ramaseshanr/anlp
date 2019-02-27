import nltk
nltk.download('stopwords')
nltk.download('gutenberg')
from nltk.probability import  FreqDist
from nltk.corpus import stopwords
import numpy as np
import pandas as pd

stop_words = set(stopwords.words('english'))

#read the corpus
words = nltk.Text(nltk.corpus.gutenberg.words('bryant-stories.txt'))
#convert to small letters
words=[word.lower() for word in words if word.isalpha() ]
words=[word.lower() for word in words if word not in stop_words ]

fDist = FreqDist(words)

#print(len(words)) #21718
#print(len(set(words))) #3688 - unique words
heading = ['Word','Frequency']
tf_list = []
for x,v in fDist.most_common(10):
    tf_list.append((x,v))
print(pd.DataFrame(tf_list,columns=heading))

heading = ['Word','Weighted Frequency']
tf_list = []

print()
for x,v in fDist.most_common(20):
    tf_list.append((x,v/len(fDist)))
print(pd.DataFrame(tf_list,columns=heading))
