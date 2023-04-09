# Pre-Trained-ASR-Models-in-Flask-API

## Intro

Here we leverage two open source ASR pre trained models, namely Whisper and Wav-2-Vec-2. These models have been trained with large datasets and have good accuracy out of the box and can be integrated easily for the process of transcripting.

WHisper:
https://github.com/openai/whisper

Wav2Vec2:
https://huggingface.co/docs/transformers/model_doc/wav2vec2

## Implementation

The pre trained models are curled to return the output transcript for given audio that is being passed as input, implemneted vis Flask API.
