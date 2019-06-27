# -*- coding: utf-8 -*-
# 
import numpy as np

#padding the iputs 
def pad_word_batch(word_batch, pad_int):
    
    max_word = max([len(word) for word in word_batch])
    return [word + [pad_int] * (max_word - len(word)) for word in word_batch]


def get_batches(sources, targets, batch_size, source_pad_int, target_pad_int):
    
    for batch_i in range(0, len(sources)//batch_size):
        start_i = batch_i * batch_size

        # Slice the right amount for the batch
        sources_batch = sources[start_i:start_i + batch_size]
        targets_batch = targets[start_i:start_i + batch_size]

        # Pad
        pad_sources_batch = np.array(pad_word_batch(sources_batch, source_pad_int))
        pad_targets_batch = np.array(pad_word_batch(targets_batch, target_pad_int))
        
        
        # Need the lengths for the _lengths parameters
        pad_targets_lengths = []
        for target in pad_targets_batch:
            pad_targets_lengths.append(len(target))

        pad_source_lengths = []
        for source in pad_sources_batch:
            pad_source_lengths.append(len(source))

        yield pad_sources_batch, pad_targets_batch, pad_source_lengths, pad_targets_lengths
        
def get_accuracy(target, logits):
    
    #Calculate accuracy
    
    max_seq = max(target.shape[1], logits.shape[1])
    if max_seq - target.shape[1]:
        target = np.pad(
            target,
            [(0,0),(0,max_seq - target.shape[1])],
            'constant')
    if max_seq - logits.shape[1]:
        logits = np.pad(
            logits,
            [(0,0),(0,max_seq - logits.shape[1])],
            'constant')

    return np.mean(np.equal(target, logits))
