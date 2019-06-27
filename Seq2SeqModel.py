# -*- coding: utf-8 -*-
#

# pandas used for reading data
import pandas as pd

#pickle for dumping(saving) data in a file
import pickle

import tensorflow as tf

#Reading dataset
dataset = pd.read_csv("transliteration.txt",delimiter = "\t",header=None,encoding='utf-8',na_filter = False)

#Splitting English words in X and Hindi words in y
X = dataset.iloc[:,0]
y = dataset.iloc[:,-1]

#importing the preprocessed data 
#The preprocessing is done in a file named Data_preprocessing
import Data_preprocessing

#source_int_text is the English words' processed vector. (Word is Converted to integer vector)
#target_int_text is the Hindi words' processed vector.
#source_vocab_to_int and  source_int_to_vocab are the English dictionaries
#target_vocab_to_int and  target_int_to_vocab are the Hindi dictionaries

source_int_text, target_int_text, source_vocab_to_int, target_vocab_to_int,source_int_to_vocab,target_int_to_vocab = Data_preprocessing.preprocess(X,y)

#encoder and decoder layers are defined in Layers 
#placeholders are defined in Model_Inputs
import Layers
import Model_Inputs


def seq2seq_model(input_data, target_data, keep_prob, batch_size,
                  target_sequence_length,
                  max_target_word_length,
                  source_vocab_size, target_vocab_size,
                  enc_embedding_size, dec_embedding_size,
                  rnn_size, num_layers, target_vocab_to_int):

#    Build the Sequence-to-Sequence model
#    :return: Tuple of (Training BasicDecoderOutput, Inference BasicDecoderOutput)
    enc_outputs, enc_states = Layers.encoding_layer(input_data, 
                                             rnn_size, 
                                             num_layers, 
                                             keep_prob, 
                                             source_vocab_size, 
                                             enc_embedding_size)
    
    dec_input = Model_Inputs.process_decoder_input(target_data, 
                                      target_vocab_to_int, 
                                      batch_size)
    
    train_output, infer_output = Layers.decoding_layer(dec_input,
                                               enc_states, 
                                               target_sequence_length, 
                                               max_target_word_length,
                                               rnn_size,
                                              num_layers,
                                              target_vocab_to_int,
                                              target_vocab_size,
                                              batch_size,
                                              keep_prob,
                                              dec_embedding_size)
    
    return train_output, infer_output



# initialising the parameters
display_step = 200

epochs = 60
batch_size = 30

rnn_size = 64
num_layers = 2

encoding_embedding_size = 50
decoding_embedding_size = 50

learning_rate = 0.001
keep_probability = 0.5


# path for saving the model
save_path = 'checkpoints/dev'


#initialising the graph
train_graph = tf.Graph()
with train_graph.as_default():
#   taking input placeholders for encoder and decoder
    input_data, targets, target_sequence_length, max_target_sequence_length = Model_Inputs.enc_dec_model_inputs()
#   taking learning rate and probability of drop out layer    
    lr, keep_prob = Model_Inputs.hyperparam_inputs()
#   compiling the model here
    train_logits, inference_logits = seq2seq_model(tf.reverse(input_data, [-1]),
                                                   targets,
                                                   keep_prob,
                                                   batch_size,
                                                   target_sequence_length,
                                                   max_target_sequence_length,
                                                   len(source_vocab_to_int),
                                                   len(target_vocab_to_int),
                                                   encoding_embedding_size,
                                                   decoding_embedding_size,
                                                   rnn_size,
                                                   num_layers,
                                                   target_vocab_to_int)
    
    training_logits = tf.identity(train_logits.rnn_output, name='logits')
    inference_logits = tf.identity(inference_logits.sample_id, name='predictions')


    masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name='masks')

    with tf.name_scope("optimization"):
        # Loss function - weighted softmax cross entropy
        cost = tf.contrib.seq2seq.sequence_loss(
            training_logits,
            targets,
            masks)

        # Optimizer
        optimizer = tf.train.AdamOptimizer(lr)

        # Gradient Clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)
        


# Split data to training and validation sets
train_source = source_int_text[batch_size:]
train_target = target_int_text[batch_size:]
valid_source = source_int_text[:batch_size]
valid_target = target_int_text[:batch_size]

# Batch Generator and Accuracy is defined in Batch_Metrics file
import Batch_Metrics

(valid_sources_batch, valid_targets_batch, valid_sources_lengths, valid_targets_lengths ) = next(Batch_Metrics.get_batches(valid_source,
                                                                                                             valid_target,
                                                                                                             batch_size,
                                                                                                             source_vocab_to_int['<PAD>'],
                                                                                                             target_vocab_to_int['<PAD>']))                                                                                                  

#Starting the the session
with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(epochs):
        for batch_i, (source_batch, target_batch, sources_lengths, targets_lengths) in enumerate(
                Batch_Metrics.get_batches(train_source, train_target, batch_size,
                            source_vocab_to_int['<PAD>'],
                            target_vocab_to_int['<PAD>'])):

            _, loss = sess.run(
                [train_op, cost],
                {input_data: source_batch,
                 targets: target_batch,
                 lr: learning_rate,
                 target_sequence_length: targets_lengths,
                 keep_prob: keep_probability})

        
            if batch_i % display_step == 0 and batch_i > 0:
                batch_train_logits = sess.run(
                    inference_logits,
                    {input_data: source_batch,
                     target_sequence_length: targets_lengths,
                     keep_prob: 1.0})

                batch_valid_logits = sess.run(
                    inference_logits,
                    {input_data: valid_sources_batch,
                     target_sequence_length: valid_targets_lengths,
                     keep_prob: 1.0})

                train_acc = Batch_Metrics.get_accuracy(target_batch, batch_train_logits)
                valid_acc = Batch_Metrics.get_accuracy(valid_targets_batch, batch_valid_logits)
                print('Epoch {:>3} Batch {:>3}/{} - Train Accuracy: {:>6.4f}, Validation Accuracy: {:>6.4f}, Loss: {:>6.4f}'
                      .format(epoch_i+1, batch_i, len(source_int_text) // batch_size, train_acc, valid_acc, loss))

    # Save Model
    saver = tf.train.Saver()
    saver.save(sess, save_path)
    print('Model Trained and saved')
    
# Save the parameters   
def save_params(params):
    with open('params.p', 'wb') as out_file:
        pickle.dump(params, out_file)

save_params(save_path)



