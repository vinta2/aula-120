import nltk
import json
import pickle
import numpy as np
import random

ignore_words=['?','!',',','.',"'s","'m"]

import tensorflow
from data_preprocessing import get_stem_words
model=tensorflow.keras.models.load_model('')

intents=json.loads(open('./intents.json').read())
words=pickle.load(open('./words.pekl','rb'))
classes=pickle.load(open('./classes.pekl','rb'))

def preprocess_user_input(user_input):
    input_wt1=nltk.word_tokenize(user_input)
    input_wt2=get_stem_words(input_wt1,ignore_words)
    input_wt2=sorted(list(set(input_wt2)))

    bag=[]
    bag_of_words=[]
    for word in words:
        if word in input_wt2:
            bag_of_words.append(1)
        else :
            bag_of_words.append(0)
    bag.append(bag_of_words)
    return np.array(bag)
def bot_class_prediction(user_input):
    inp=preprocess_user_input(user_input)
    prediction=model.predict(inp)
    predicted_class_label=np.argmax(prediction[0])
    return predicted_class_label
def bot_response(user_input):
    predicted_class_label=bot_class_prediction(user_input)
    predicted_class=classes[predicted_class_label]
    for intent in intents['intents']:
        if intent['tag']==predicted_class:
            bot_response=random.choice(intent['responses'])
            return bot_response
print("oi,eu sou vitor,como posso ajudar?")
while True:
    user_input=input("digite sua mensagem aqui:")
    print("entrada do usuario:",user_input)

    response=bot_response(user_input)
    print("resposta do bot:",response)
