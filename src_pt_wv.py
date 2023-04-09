#!/usr/bin/env python
# coding: utf-8

# In[1]:

import torch
from flask import Flask, abort, request
import whisper
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import torchaudio
from tempfile import NamedTemporaryFile

# Load the Whisper model:
#model = whisper.load_model('base')
def transcribe(input_audio):
    tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-large-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
    input_values = tokenizer(input_audio, return_tensors="pt", padding="longest").input_values

    logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.batch_decode(predicted_ids)

    return(transcription)

app = Flask(__name__)

@app.route('/', methods=['POST'])
def handler():
    if not request.files:
        # If the user didn't submit any files, return a 400 (Bad Request) error.
        abort(400)

    results = []
    for filename, handle in request.files.items():

        temp = NamedTemporaryFile()
        handle.save(temp)

        result = transcribe(temp)
        
        results.append({
            'transcript': result['text'],
        })


    # This will be automatically converted to JSON.
    return {'results': results}
