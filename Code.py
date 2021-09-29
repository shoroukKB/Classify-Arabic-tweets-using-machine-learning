#-------------import lablary------------------------------------------------
import numpy as np
import pandas as pd  # for loading the data from files and there are other options
import copy
import pyarabic.araby as araby
import pyarabic.number as number
from collections import Counter
from sklearn import preprocessing as p
from sklearn.model_selection import train_test_split
import re
import string
import preprocessor as p #to clear data
from pyarabic.araby import strip_tashkeel
import sys
########################################################################
#--------------load data -----------------------------------------------
dataset = pd.read_csv("250.csv")
#change colom names
#dataset=dataset.rename(columns={'Classfication': 'Class', 'Tweet Text': 'Tweets'})
dataset=dataset.rename(columns={'Tweet Text': 'Tweets'})

#drop all rows that have any NaN values
dataset.dropna()     


#data cleaining 
#-------------- remove emojis and links---------------------------#
emoji_pattern  = re.compile(
    "(["
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F700-\U0001F77F"  # alchemical symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U00002702-\U000027B0"  # Dingbats
    "])", flags=re.UNICODE
  )


tweets=dataset['Tweets']

#set of puntionation mark 
exclude = set(string.punctuation)

#so clean data final data(tweets) list
so_clean=[]

#walk throght each tweet 
for x,j in tweets.iteritems():
    tweet=j
    p.set_options(p.OPT.URL, p.OPT.HASHTAG, p.OPT.MENTION, p.OPT.EMOJI, p.OPT.SMILEY,p.OPT.SMILEY)
    cleaned_tweeet = p.clean(tweet)
    cleaned_tweeet=emoji_pattern.sub(r'', cleaned_tweeet)
    
    # strip harakat
    cleaned_tweeet= strip_tashkeel(cleaned_tweeet)
    
    #remove arabic stoppet word and some spical sybols not included in emoje range
    unnessary_symbols =[" انك " ," إنك "," الا "," ألا "," عن "," فيا "," امين "," قول "," هذا "," يا "," ليش ",
        " على "," انا ","مافي"," من "," يلي "," انو ",  " بس " , " ياله "," ابن "," انت "," فيه ",
        " فية "," أبدا "," أجل "," أخو ","أحد" ," في "," لا "," هلا ","↪","⭕","⤵","l","\u200d"]
    for i in range(len(unnessary_symbols)):
          cleaned_tweeet=cleaned_tweeet.replace(unnessary_symbols[i],' ')
    
         
    #remove punctionation marks 
    cleaned_tweeet = ''.join(ch for ch in cleaned_tweeet if ch not in exclude)
    n = filter(lambda x: True if x==' ' else x not in string.printable , cleaned_tweeet)

    #remove numbers
    cleaned_tweeet = ''.join([i for i in cleaned_tweeet if not i.isdigit()])
   

    #remove english text
    cleaned_tweeet = re.sub(r'\s*[A-Za-z]+\b', '' , cleaned_tweeet)
    cleaned_tweeet = cleaned_tweeet.rstrip()

    #remove hamza
    cleaned_tweeet = re.sub(u'ء', '' , cleaned_tweeet)
    cleaned_tweeet = cleaned_tweeet.rstrip()

    #remove dot
    cleaned_tweeet = re.sub(u'•', '' , cleaned_tweeet)
    cleaned_tweeet = cleaned_tweeet.rstrip()
    
    #split text with only one space 
    cleaned_tweeet=' '.join(cleaned_tweeet.split())   
    so_clean.append(cleaned_tweeet)
    
print (so_clean)    
       
#--------------------drop duplicate after cleaining-------------------------------#
#sorting by first name 
dataset.sort_values("Tweets", inplace = True) 
#print (dataset)

# dropping ALL duplicte values 
dataset.drop_duplicates(subset ="Tweets", keep = False, inplace = True)
 
#-----------------------bag of word feature extraction---------------#
#impot labraries
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()

#split each word in the modle and put it into array recording to number of replicated
X = vectorizer.fit_transform(so_clean)
ss=X.toarray()
print(ss.shape)
#print(type(ss))
#print (so_clean)

# concat main data fram with the number arry which represent features 
dataset=pd.concat([dataset, pd.DataFrame(ss)], axis=1)

#drop old colom (not clean and not clean represented as text)
dataset=dataset.drop(['Tweets'],axis='columns')
 

#-----------------------------------------data sprate and spliting----------------------------#
#sperate target and fetures  
var=dataset.columns.values.tolist()
y=dataset["Classfication"]
x=[i for i in var if i not in ['Classfication']]
x=dataset[x]

#spilit the data with ratio 30 for learining and 70 for testing
 
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=0)

# validate data to chose best acuracy from naive base and svm models.
X_train, X_val, y_train, y_val  = train_test_split(X_train, y_train, test_size=0.30, random_state=0)
    
#--------------------------------------------------SVM MODLE---------------------------------------#
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC

#grid search optimal parameters
param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
grid.fit(X_train,y_train)
#optimal parameters after run=C=100, gamma=0.001, kernel=sigmoid --so it is hard margin-- 
best=grid.best_estimator_

#svm modle
SVM = SVC(kernel='sigmoid', random_state=0, gamma=0.001, C=100) 
SVM.fit(X_train, y_train) 
print('Accuracy of our SVM model on the training data is {0:.2f} out of 1'.format(SVM.score(X_train, y_train))) 
print('Accuracy of our SVM model on the test data is {0:.2f} out of 1'.format(SVM.score(X_val, y_val)))

#report preformance for validate data using svm
print("Prformace for SVM modle is :\n")
estimated_y=SVM.predict(X_val)
print ("The confusion_matrix is ")
print(confusion_matrix(y_val, estimated_y))
print ("\nclassification report is ")
print(classification_report(y_val, estimated_y))

#----------------------------------NAIVE BASE MODLE---------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

#naive base modle
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_val)
print("\n\nPrformace for NEAIVE BASE modle is :\n")
print ("The confusion_matrix is ")
print(confusion_matrix(y_val, y_pred))
print ("\nclassification report is ")
print(classification_report(y_val, y_pred))

#--------------------------final modle chosen base on validate data testing------------------------------
#final model
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=0)
print("-----------------------------------------------------------------------------")
cc = GaussianNB()
y_pr = cc.fit(X_train, y_train).predict(X_test)
print("\n\nPrformace for NEAIVE BASE FOR FINAL  modle is :\n")
print ("The confusion_matrix is ")
print(confusion_matrix(y_test, y_pr))
print ("\nclassification report is ")
print(classification_report(y_test, y_pr))





