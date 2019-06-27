# -*- coding: utf-8 -*-


import tensorflow as tf
import pickle
import Data_preprocessing

#loading the saved parameters
def load_params():
    with open('params.p', mode='rb') as in_file:
        return pickle.load(in_file)

#getting the source and target vocabuaries
_, (source_vocab_to_int, target_vocab_to_int), (source_int_to_vocab, target_int_to_vocab) = Data_preprocessing.load_preprocess()

load_path = load_params()



batch_size = 30

#converting the words to vectors of integers
def word_to_seq(word, vocab_to_int):
    results = []
    for word in list(word):
        if word in vocab_to_int:
            results.append(vocab_to_int[word])
        else:
            results.append(vocab_to_int['<UNK>'])
            
    return results

#taking user input for prediction
print("\n Enter word to be transliterated:")
transliterate_word = input().lower()

transliterate_word = word_to_seq(transliterate_word, source_vocab_to_int)

#initialising the graph
loaded_graph = tf.Graph()

#initialising the session
with tf.Session(graph=loaded_graph) as sess:
        
    # Load saved model
    loader = tf.train.import_meta_graph(load_path + '.meta')
    
    #tf.train.Saver.restore(sess,load_path)
    loader.restore(sess, load_path)

#providing placeholder names from the loaded graph
    input_data = loaded_graph.get_tensor_by_name('input:0')
    logits = loaded_graph.get_tensor_by_name('predictions:0')
    target_sequence_length = loaded_graph.get_tensor_by_name('target_sequence_length:0')
    keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')

#transliterating the given word
    transliterate_logits = sess.run(logits, {input_data: [transliterate_word]*batch_size,
                                         target_sequence_length: [len(transliterate_word)]*batch_size,
                                         keep_prob: 1.0})[0]

print('Input')
print('  Word Ids:      {}'.format([i for i in transliterate_word]))
print('  English Word: {}'.format([source_int_to_vocab[i] for i in transliterate_word]))

print('\nPrediction')
print('  Word Id:      {}'.format([i for i in transliterate_logits]))

#showing the output
output = ""
for i in transliterate_logits:
        if target_int_to_vocab[i]!= '<EOS>':
                output = output + target_int_to_vocab[i]
print('  Hindi Word:      {}'.format(output))