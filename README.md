# Machine Transliteration ( English to Hindi ) Tensorflow

In this project, we have built a character level language model for [transliterating](https://en.wikipedia.org/wiki/Transliteration) English text into Hindi. It is implemented using TensorFlow. 
Transliteration is the phonetic translation of the words in a source language ( here English ) into equivalent words in a target language ( here Hindi ). It preserves the pronunciation of the words.  
This project has been inspired by the NMT model developed by [deep-diver]( https://github.com/deep-diver/EN-FR-MLT-tensorflow ).

### Note: Refer to [GUIDE.md](/GUIDE.md) for a brief on code implementation.

## Overview of the Architecture
Transliteration being a type of many to many problem, we built a encoder-decoder model in TensorFlow. The objective of the model is Transliterating English text to Hindi script.

* **Dataset:** We have used [FIRE 2013](http://cse.iitkgp.ac.in/resgrp/cnerg/qa/fire13translit/index.html) dataset to train the model. FIRE dataset is useful for transliteration tasks, the one we used contains 30,823 word transliteration pairs of English to Hindi.
* **Exploring the data:**
The data has been stored separately into two variables; source text and target text i.e. English and Hindi respectively



## Preprocessing: 
The preprocessing contains three important steps which include:
*	**Creating lookup tables:** 
We made dictionaries (i.e. mapping tables) of character to corresponding character id and vice versa for both the source and target characters (vocabulary). Now we have 4 lookup tables.
*	**Text to character ids:**
 We converted each character in the list of words to the corresponding index with the help of look-up tables
The pre-processed data is saved to the external file.

## Building the Neural Network:
We then created a sequence to sequence model i.e. **Encoder-Decoder layers**. 
1.	**Inputs to the encoding layer:**
    * `enc_dec_model_inputs()` : we have defined a function to create placeholders for encoder-decoder inputs.
    * `hyperparam_inputs()` : to create placeholders for the hyper parameters that are needed later, `lr_rate` (learning rate) and `keep_prob` (probability of dropouts).

2.	**Encoder:**
It contains two parts: Embedding layer & RNN layer. The former converts each character in the word with the number of features specified as `encoding_embedding_size`. The later part being the RNN layer, we have used _LSTM cells_ stacked together after dropout technique. 

3.	**Decoder:**
Decoding model mainly comprises of two phases, _Training_ and _Inference_. 
Both of them share the same architecture and parameters, but the difference comes in feeding the shared model.
 * Initially we preprocess the target label data for the training phase, i.e. we add a special token `<GO>` in front of all target data to imply the start of transliteration.
 * While passing the embedding layer to the decoder, we cannot use `tf.contrib.layers.embed_sequence` like in encoder. The reason being, we need to pass the embedding layer to both training and inference phases and `tf.contrib.layers.embed_sequence` embeds the prepared dataset before running, but in inference phase we need the dynamic embedding capability.
 * In decoder-training-phase, the embeded input is passed as input in each time step. Whereas in decoder-inference-phase, the the output of the previous time-step is dynamically passed over embedding parameters and fed to the next time step.
`tf.variable.scope()` is used for the sharing of parameters and variables between training and inference processes since they both share the same architecture. 

## To build the seq2seq model:
Here, the previously created layers, `encoding_layer`, `process_decoder_input` and `decoding_layer` are put together to build the full-fledged Sequence to Sequence model.

## Build the Graph:
1. **Cost Function:**
     `tf.contrib.seq2seq.sequence_loss` is used for calculating loss function i.e., a weighted softmax cross entropy loss function. Weights are explicitly provided as an argument, and it can be created by `tf.sequence_mask`. 

2. **Optimizer:**
We used _Adam optimizer_ (`tf.train.AdamOptimizer`) with specified learning rate. 

3. **Gradient Clipping:**
`tf.clip_by_value` is used to do the gradient clipping to overcome exploding gradient.

## Train:
We defined get_accuracy function to compute train and validation accuracy.
 
![Training and Validation accuracy of the model over 60 epochs](/Screenshot_accuracy.JPG)

## Save parameters:
We then saved the `batch_size` and `save_path` parameters for inference.

## Transliterate:
We then defined a function `word_to_seq` to do the transliteration of input words by the user.
 
![Sample Prediction](/ExampleScreenshot.JPG) .




 


