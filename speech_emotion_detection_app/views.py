from django.shortcuts import render

from .forms import *
from .models import Audio
from django.views.generic.edit import CreateView
from django.contrib.messages.views import SuccessMessageMixin
from django.views import View
from .forms import AudioForm
from .preprocess import preprocess_audio
from keras.models import model_from_json
import json
import numpy
import speech_recognition as sr
#import pandas as pd
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
#import contractions
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
# Create your views here.

emotions = {
    0 : 'Angry',
    1 : 'Excited',
    2 : 'Neutral',
    3 : 'Sad'
}


def preprocess_text(input_text):

    #Removing unwanted symbols
    
    new_string = ""
    for word in input_text.replace("\x92", "'").replace("\x85", " ").replace("\x97", " ").replace("\x91", " ").split(" "):
        new_string+= word
        new_string+= " "
    input_text = new_string
    
    # Removing contractions
    #for i in range(len(input_text)):
    #input_text = contractions.fix(input_text)

    # Removing punctuation and stop words.
    stop_words = stopwords.words('english')

    #for i in range(0, len(input_text)):
    review = re.sub('[^a-zA-Z]', ' ', input_text) 
    review = review.lower()
    review = review.split()
        
    review = [word for word in review if not word in stop_words] 
    review = ' '.join(review)
    input_text = review
    print(input_text)
    X = []
    X.append(input_text)
    # using keras tokenizer here
    token = Tokenizer(num_words=None)
    max_len = 70
    a = X
    print(a)
    token.fit_on_texts(a)
    input_text_seq = token.texts_to_sequences(X)
    print(input_text_seq)
    # zero pad the sequences
    input_text_pad = pad_sequences(input_text_seq, maxlen=max_len)
    print(input_text_pad)
    #word_index = token.word_index

    #with open('Matrix\embedding_matrix.pkl','rb') as f:
    #    embedding_matrix = pickle.load(f)

    return input_text_pad




def predictOutput(audio):
    r = sr.Recognizer()
    input_text = ""
    with sr.AudioFile(audio) as source:
        audio_text = r.listen(source)
        # recoginize_() method will throw a request error if the API is unreachable, hence using exception handling
        try:
            # using google speech recognition
            input_text = r.recognize_google(audio_text)
            #print('Converting audio transcripts into text ...')
            print(input_text)
        except:
            print('Sorry.. run again...')

    input_file = preprocess_audio(audio)
    X = numpy.array(input_file)
    input_text_pad = preprocess_text(input_text)
    #X = numpy.array(input_file)
    #print(input_file)
    #print(X.shape)
    X_text = []
    X_text.append(input_text_pad)
    #print(input_file)
    #print(X.shape)
    json_file = open('DeepLearningModel/iemocap_combined_lstm_lstm_128_64bs.json', 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    # load weights into new model
    model.load_weights("DeepLearningModel/iemocap_combined_lstm_lstm_128_64bs.h5")
    predictions = model.predict([X_text, X])
    labels = [0,1,2,3]
    output_emotion = labels[numpy.argmax(predictions)]
    return emotions[output_emotion]

class Upload_audio(SuccessMessageMixin, CreateView):
    success_url = '/view'
    success_message = 'Your audio has been uploaded'
    error_message = 'No audio file'
    form_class = AudioForm
    template_name = 'index.html'

class ViewAudio(View):
    def get(self, request):
        audios  = Audio.objects.latest('id')
        #print(audios.audio_file.path)
        emotion = predictOutput(audios.audio_file.path)
        context = { 'audios' : audios , 'emotion' : emotion }
        return render(request, "index.html", context)