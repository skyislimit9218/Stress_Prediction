import pickle

#cv = CountVectorizer()
filename ="model_save.sav"
loaded_model = pickle.load(open(filename, 'rb'))
cv= pickle.load(open("vectorizer.pickle", 'rb'))
user = input("Enter a Text: ")
data = cv.transform([user]).toarray()
output = loaded_model.predict(data)
print(output)
