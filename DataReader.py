import sys
import numpy
import numpy as np
from tempfile import TemporaryFile
import random

import timeit
import cPickle
import json

from conf import *

MENTION_TYPES = {
    "PRONOMINAL": 0,
    "NOMINAL": 1,
    "PROPER": 2,
    "LIST": 3
}
MENTION_NUM, SENTENCE_NUM, START_INDEX, END_INDEX, MENTION_TYPE, CONTAINED = 0, 1, 2, 3, 4, 5

DIR = "/home/qingyu/data/kevin/"
embedding_file = DIR+"features/mention_data/word_vectors.npy"

span_dimention = 5*50
embedding_dimention = 50
embedding_size = 34275
word_embedding_dimention = 9*50

numpy.set_printoptions(threshold=numpy.nan)
random.seed(args.random_seed)

class DataGnerater():
    def __init__(self,file_name):

        doc_path = "/home/qingyu/data/kevin/features/doc_data/%s/"%file_name
        pair_path = "/home/qingyu/data/kevin/features/mention_pair_data/%s/"%file_name
        mention_path = "/home/qingyu/data/kevin/features/mention_data/%s/"%file_name

        gold_path = "/home/qingyu/data/kevin/gold/"+file_name.split("_")[0]
        # read gold chain
        self.gold_chain = {}
        gold_file = open(gold_path)
        golds = gold_file.readlines()
        for item in golds:
            gold = json.loads(item) 
            self.gold_chain[int(gold.keys()[0])] = list(gold[gold.keys()[0]])

    #embedding_matrix = numpy.load(embedding_file) 

    ## for mentions
        self.mention_spans = numpy.load(mention_path+"msp.npy")
        self.mention_word_index = numpy.load(mention_path+"mw.npy") 
        self.mention_feature = numpy.load(mention_path+"mf.npy")
        self.mention_id = numpy.load(mention_path+"mid.npy")[:,0]
        self.mention_did = numpy.load(mention_path+"mdid.npy")[:,0]
        self.mention_num = numpy.load(mention_path+"mnum.npy")[:,0]

        self.mention_pair_feature = numpy.load(mention_path+"yqy.npy",mmap_mode='r')
        #self.mention_pair_feature = numpy.lib.format.open_memmap(mention_path+"pi.npy")

    ## for pairs
        #self.pair_feature = numpy.load(pair_path + 'pf.npy')
        self.pair_coref_info = numpy.load(pair_path + "y.npy")    
        self.pair_index = numpy.load(pair_path + "pi.npy")
        self.pair_mention_id = numpy.load(pair_path + "pmid.npy")

    ## for docs
        self.document_features = numpy.load(doc_path + 'df.npy')
        self.doc_pairs = numpy.load(doc_path + 'dpi.npy') # each line is the pair_start_index -- pair_end_index
        self.doc_mentions = numpy.load(doc_path + 'dmi.npy') # each line is the mention_start_index -- mention_end_index


    def generater(self,shuffle=False):

        # build training data  
        doc_index = range(self.doc_pairs.shape[0])
        if shuffle:
            numpy.random.shuffle(doc_index) 

        done_num = 0
        total_num = self.doc_pairs.shape[0]
        estimate_time = 0.0
        for did in doc_index:
            start_time = timeit.default_timer() 
            ps, pe = self.doc_pairs[did]
            ms, me = self.doc_mentions[did]
            
            done_num += 1

            doc_mention_sizes = me - ms

            document_feature = self.document_features[did] 

            # build training data for each doc
            mention_span_indoc = self.mention_spans[ms:me]
            mention_word_index_indoc = self.mention_word_index[ms:me]
            mention_feature_indoc = self.mention_feature[ms:me]
            mention_num_indoc = self.mention_num[ms:me]
            mention_id_real = self.mention_id[ms:me]
        
            mention_feature_list = []
            for mf in mention_feature_indoc:
                mention_feature_list.append(get_mention_features(mf,me-ms,document_feature))
            mention_feature_list = numpy.array(mention_feature_list)
       
            mention_pair_feature_indoc = self.mention_pair_feature[ps:pe]    

            pair_coref_indoc = self.pair_coref_info[ps:pe].astype(int)
            pair_mention_id_indoc = self.pair_mention_id[ps:pe]

            target_for_each_mention = []
            mention_id_for_each_mention = []
            pair_feature_for_each_mention = []
            st = 0
            for i in range(len(mention_feature_list)):
                target = pair_coref_indoc[st:st+i] # if mention is index r, it has r antecedents
                target_for_each_mention.append(target)

                pair_feature_current_mention = mention_pair_feature_indoc[st:st+i]
                pair_feature_for_each_mention.append(pair_feature_current_mention)

                mention_ids = pair_mention_id_indoc[st:st+i]
                
                this_mention_id = mention_id_real[i]
                candidates_id = [] 
                if len(mention_ids) > 0:
                    candidates_id = mention_ids[:,1].tolist()
                mention_id_for_each_mention.append((did,this_mention_id,candidates_id))
                st = st+i
           
            inside_index = range(len(mention_feature_list))
            if shuffle:
                numpy.random.shuffle(inside_index)

            for i in inside_index:
                ana_word_index = mention_word_index_indoc[i]
                ana_span = mention_span_indoc[i]
                ana_feature = mention_feature_list[i]
            
                candi_word_index = mention_word_index_indoc[:i]
                candi_span = mention_span_indoc[:i]

                pair_feature_array = pair_feature_for_each_mention[i]
             
                this_thrainig_data = (ana_word_index,ana_span,ana_feature,candi_word_index,candi_span,pair_feature_array,target_for_each_mention[i],mention_id_for_each_mention[i])
                ## mention_id_for_each_mention: list, each item is like : (doc_id,current_mention_id, candidate_id)

                doc_end = False
                if i == inside_index[-1]:
                    doc_end = True

                yield this_thrainig_data,doc_end

            end_time = timeit.default_timer()
            estimate_time += (end_time-start_time)
            EST = total_num*estimate_time/float(done_num)
            print >> sys.stderr, "Total use %.3f seconds for doc %d with %d mentions (%d/%d) -- EST:%f , Left:%f"%(end_time-start_time,did,me - ms,done_num,total_num,EST,EST-estimate_time)
    
def get_mention_features(mention_features, doc_mention_size,document_features):
    features = numpy.array([])
    features = numpy.append(features,one_hot(mention_features[MENTION_TYPE], 4)) 
    features = numpy.append(features,distance(np.subtract(mention_features[END_INDEX] - mention_features[START_INDEX], 1)))
    features = numpy.append(features,float(mention_features[MENTION_NUM])/ float(doc_mention_size))
    features = numpy.append(features,mention_features[CONTAINED])
    features = numpy.append(features,document_features)
    return features

def get_distance_features(m1, m2):
    dis_f = numpy.array(int((m2[SENTENCE_NUM] == m1[SENTENCE_NUM]) & (m1[END_INDEX] > m2[START_INDEX])))
    dis_f = numpy.append(dis_f,distance(m2[SENTENCE_NUM] - m1[SENTENCE_NUM]))
    dis_f = numpy.append(dis_f,distance(np.subtract(m2[MENTION_NUM] - m1[MENTION_NUM], 1)))
    return dis_f


def one_hot(a, n): 
    oh = np.zeros(n) 
    oh[a] = 1 
    return oh

def distance(a):
    d = np.zeros(11)
    d[a == 0, 0] = 1
    d[a == 1, 1] = 1
    d[a == 2, 2] = 1
    d[a == 3, 3] = 1
    d[a == 4, 4] = 1
    d[(5 <= a) & (a < 8), 5] = 1
    d[(8 <= a) & (a < 16), 6] = 1
    d[(16 <= a) & (a < 32), 7] = 1
    d[(a >= 32) & (a < 64), 8] = 1
    d[a >= 64, 9] = 1
    d[10] = np.clip(a, 0, 64) / 64.0
    return d

if __name__ == "__main__":
    data,doc_end = DataGnerater("test_reduced")   
    for t in data.generater():
        pass
