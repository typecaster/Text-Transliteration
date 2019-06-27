# -*- coding: utf-8 -*-
#

import copy
import pickle

CODES = {'<PAD>': 0, '<EOS>': 1, '<UNK>': 2, '<GO>': 3 }

def create_lookup_table(vocab):
    # make a list of unique characters
    
    vocab = set(list(vocab))
    vocab_to_int = copy.copy(CODES)
    for v_i, v in enumerate(vocab, len(CODES)):
            vocab_to_int[v] = v_i
            
    int_to_vocab = {v_i: v for v, v_i in vocab_to_int.items()}
        
    return vocab_to_int, int_to_vocab




def text_to_ids(source_words, target_words, source_vocab_to_int, target_vocab_to_int):
    
        #1st, 2nd args: raw string text to be converted
        #3rd, 4th args: lookup tables for 1st and 2nd args respectively
    
        #return: A tuple of lists (source_id_text, target_id_text) converted
    
    # empty list of converted words
    source_text_id = []
    target_text_id = []
    
    max_source_word_length = max([len(word) for word in source_words])
    max_target_word_length = max([len(word) for word in target_words])
    
    # iterating through each word (# of words in source&target is the same)
    for i in range(len(source_words)):
        # extract words one by one
        source_word = source_words[i]
        target_word = target_words[i]
        
        # make a list of characters (extraction) from the chosen word
        source_tokens = list(source_word)
        target_tokens = list(target_word)
        
        # empty list of converted words to index in the chosen word
        source_token_id = []
        target_token_id = []
        
        for index, token in enumerate(source_tokens):
                source_token_id.append(source_vocab_to_int[token])
        
        for index, token in enumerate(target_tokens):
                target_token_id.append(target_vocab_to_int[token])
                
        # put <EOS> token at the end of the chosen target word
        # this token suggests when to stop creating a sequence
        target_token_id.append(target_vocab_to_int['<EOS>'])
            
        # add each converted words in the final list
        source_text_id.append(source_token_id)
        target_text_id.append(target_token_id)
    
    return source_text_id, target_text_id


def preprocess(source_text, target_text):


    Englishvocab = 'abcdefghijklmnopqrstuvwxyz'    
    Hindivocab = 'ँंॉॆॊॏऺऻॎःािीुूेैोौअआइईउऊएऐओऔकखगघचछजझटठडढणतथदधनपफबभमयरलवशषसहज्ञक्षश्रज़रफ़ड़ढ़ख़क़ग़ळृृ़़ऑ'
    
    # create lookup tables for English and Hindi data
    source_vocab_to_int, source_int_to_vocab = create_lookup_table(Englishvocab)
    target_vocab_to_int, target_int_to_vocab = create_lookup_table(Hindivocab)

    # create list of words whose characters are represented in index
    source_text, target_text = text_to_ids(source_text, target_text, source_vocab_to_int, target_vocab_to_int)
    
     # Save data for later use
    pickle.dump((
        (source_text, target_text),
        (source_vocab_to_int, target_vocab_to_int),
        (source_int_to_vocab, target_int_to_vocab)), open('preprocess.p', 'wb'))
    
    return source_text,target_text,source_vocab_to_int,target_vocab_to_int,source_int_to_vocab,target_int_to_vocab

# function to load the preprocessed data
def load_preprocess():
    with open('preprocess.p', mode='rb') as in_file:
        return pickle.load(in_file)
