from flask import Flask, request
app = Flask(__name__)

from utils import * 
import pickle 
import time 
from tqdm import tqdm
import argparse
import numpy as np 
import pickle 
import tensorflow as tf 
from tensorflow.contrib import learn 
from tflearn.data_utils import to_categorical, pad_sequences

parser = argparse.ArgumentParser(description="Test URLNet model")
sess = None
# data args
default_max_len_words = 200
max_len_words = default_max_len_words
default_max_len_chars = 200
max_len_chars = default_max_len_chars
default_max_len_subwords = 20
max_len_subwords = default_max_len_subwords
default_delimit_mode = 1
delimit_mode = default_delimit_mode
subword_dict_dir = "runs/10000/subwords_dict.p"
word_dict_dir = "runs/10000/words_dict.p"
char_dict_dir = "runs/10000/chars_dict.p"

# model args 
default_emb_dim = 32
emb_dim = default_emb_dim
default_emb_mode = 1
emb_mode = default_emb_mode
# test args 
default_batch_size = 1
batch_size = default_batch_size

# log args 
output_dir = "runs/10000/"
checkpoint_dir = "runs/10000/checkpoints/"


ngram_dict = pickle.load(open(subword_dict_dir, "rb")) 
print("Size of subword vocabulary (train): {}".format(len(ngram_dict)))
word_dict = pickle.load(open(word_dict_dir, "rb"))
print("size of word vocabulary (train): {}".format(len(word_dict)))

######################## EVALUATION ########################### 

def test_step(x, emb_mode):
    p = 1.0
    if emb_mode == 1: 
        feed_dict = {
            input_x_char_seq: x[0],
            dropout_keep_prob: p}  
    elif emb_mode == 2: 
        feed_dict = {
            input_x_word: x[0],
            dropout_keep_prob: p}
    elif emb_mode == 3: 
        feed_dict = {
            input_x_char_seq: x[0],
            input_x_word: x[1],
            dropout_keep_prob: p}
    elif emb_mode == 4: 
        feed_dict = {
            input_x_word: x[0],
            input_x_char: x[1],
            input_x_char_pad_idx: x[2],
            dropout_keep_prob: p}
    elif emb_mode == 5:  
        feed_dict = {
            input_x_char_seq: x[0],
            input_x_word: x[1],
            input_x_char: x[2],
            input_x_char_pad_idx: x[3],
            dropout_keep_prob: p}
    preds, s = sess.run([predictions, scores], feed_dict)
    return preds, s

checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
graph = tf.Graph() 
with graph.as_default(): 
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_conf.gpu_options.allow_growth=True 
    sess = tf.Session(config=session_conf)
    with sess.as_default(): 
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file) 
        
        if  emb_mode in [1, 3, 5]: 
            input_x_char_seq = graph.get_operation_by_name("input_x_char_seq").outputs[0]
        if emb_mode in [2, 3, 4, 5]:
            input_x_word = graph.get_operation_by_name("input_x_word").outputs[0]
        if emb_mode in [4, 5]:
            input_x_char = graph.get_operation_by_name("input_x_char").outputs[0]
            input_x_char_pad_idx = graph.get_operation_by_name("input_x_char_pad_idx").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0] 

        predictions = graph.get_operation_by_name("output/predictions").outputs[0]
        scores = graph.get_operation_by_name("output/scores").outputs[0]
         
        

    @app.route("/api/predict/", methods=["POST"])
    def hello():
        link = request.form["link"]
        urls = [link]
        x, word_reverse_dict = get_word_vocab(urls, max_len_words) 
        word_x = get_words(x, word_reverse_dict, delimit_mode, urls) 

        ngramed_id_x, worded_id_x = ngram_id_x_from_dict(word_x, max_len_subwords, ngram_dict, word_dict) 
        chars_dict = pickle.load(open(char_dict_dir, "rb"))          
        chared_id_x = char_id_x(urls, chars_dict, max_len_chars)    

        print("Number of testing urls: {}".format(len(urls)))
        if emb_mode == 1: 
            batches = batch_iter(list(chared_id_x), batch_size, 1, shuffle=False) 
        elif emb_mode == 2: 
            batches = batch_iter(list(worded_id_x), batch_size, 1, shuffle=False) 
        elif emb_mode == 3: 
            batches = batch_iter(list(zip(chared_id_x, worded_id_x)), batch_size, 1, shuffle=False)
        elif emb_mode == 4: 
            batches = batch_iter(list(zip(ngramed_id_x, worded_id_x)), batch_size, 1, shuffle=False)
        elif emb_mode == 5: 
            batches = batch_iter(list(zip(ngramed_id_x, worded_id_x, chared_id_x)), test.batch_size, 1, shuffle=False)    
        all_predictions = []
        all_scores = []
        
        
        nb_batches = int(len(urls) / batch_size)
        nb_batches = 1 
        print("Number of batches in total: {}".format(nb_batches))
        it = tqdm(range(nb_batches), desc="emb_mode {} delimit_mode {} test_size {}".format(emb_mode, delimit_mode, len(urls)), ncols=0)
        for idx in it:
        #for batch in batches:
            batch = next(batches)

            if emb_mode == 1: 
                x_char_seq = batch 
            elif emb_mode == 2: 
                x_word = batch 
            elif emb_mode == 3: 
                x_char_seq, x_word = zip(*batch) 
            elif emb_mode == 4: 
                x_char, x_word = zip(*batch)
            elif emb_mode == 5: 
                x_char, x_word, x_char_seq = zip(*batch)            

            x_batch = []    
            if emb_mode in[1, 3, 5]: 
                x_char_seq = pad_seq_in_word(x_char_seq, max_len_chars) 
                x_batch.append(x_char_seq)
            if emb_mode in [2, 3, 4, 5]:
                x_word = pad_seq_in_word(x_word, max_len_words) 
                x_batch.append(x_word)
            if emb_mode in [4, 5]:
                x_char, x_char_pad_idx = pad_seq(x_char, max_len_words, max_len_subwords, emb_dim)
                x_batch.extend([x_char, x_char_pad_idx])
            
            batch_predictions, batch_scores = test_step(x_batch, emb_mode)
            print("batch_predictions: {}".format(batch_predictions))
            print("batch_scores: {}".format(batch_scores))        
        return str(batch_predictions[0])

