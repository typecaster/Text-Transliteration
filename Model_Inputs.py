# -*- coding: utf-8 -*-
#

import tensorflow as tf
def enc_dec_model_inputs():
#   English Inputs
    inputs = tf.placeholder(tf.int32, [None, None], name='input')
    
#   Hindi Inputs
    targets = tf.placeholder(tf.int32, [None, None], name='targets') 

#   Length of Hindi words
    target_sequence_length = tf.placeholder(tf.int32, [None], name='target_sequence_length')
    
    #computes the maximum of all elements across the dimension
    max_target_len = tf.reduce_max(target_sequence_length)    
    
    return inputs, targets, target_sequence_length, max_target_len

def hyperparam_inputs():
    
    #learning rate
    lr_rate = tf.placeholder(tf.float32, name='lr_rate')
    
    #probability of dropout
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    
    return lr_rate, keep_prob

def process_decoder_input(target_data, target_vocab_to_int, batch_size):

#    Preprocess target data for encoding
#    :return: Preprocessed target data
        
    # get '<GO>' id
    go_id = target_vocab_to_int['<GO>']
    
    #Take the batch of Hindi target
    #tf.strided_slice(input_,begin,end,strides=None)
    after_slice = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    
    #Concat every word of the Hindi target with <GO> id 
    after_concat = tf.concat( [tf.fill([batch_size, 1], go_id), after_slice], 1)
    
    return after_concat
