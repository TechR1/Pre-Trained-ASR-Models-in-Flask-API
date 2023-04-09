#!/usr/bin/env python
# coding: utf-8

# In[1]:

import torch
from flask import Flask, abort, request
import whisper
from denoiser import pretrained
from denoiser.dsp import convert_audio
import torchaudio
from tempfile import NamedTemporaryFile

# Load the Whisper model:
#model = whisper.load_model('base')
model = whisper.load_model('tiny')

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

        result = model.transcribe(temp)
        
        results.append({
            'transcript': result['text'],
        })


    # This will be automatically converted to JSON.
    return {'results': results}
