import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
import pickle

#cv = CountVectorizer()
filename ="model_save.sav"
loaded_model = pickle.load(open(filename, 'rb'))
cv= pickle.load(open("vectorizer.pickle", 'rb'))
user = input("Enter a Text: ")
data = cv.transform([user]).toarray()
output = loaded_model.predict(data)
print(output)
