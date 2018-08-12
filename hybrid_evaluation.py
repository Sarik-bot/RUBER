__author__ = 'liming-vie'

import os
from referenced_metric import Referenced
from unreferenced_metric import Unreferenced
import argparse


class Hybrid():
    def __init__(self,
            data_dir,
            frword2vec,
            fqembed,
            frembed,            
            qmax_length=200,
            rmax_length=200,
            training=False,
            ref_method='max_min',
            gru_units=128, mlp_units=[256, 512, 128]
        ):

        self.ref=Referenced(data_dir, frword2vec, ref_method)
        self.unref=Unreferenced(qmax_length, rmax_length,
                os.path.join(data_dir,fqembed),
                os.path.join(data_dir,frembed),  
                gru_units, mlp_units,
                train_dir=train_dir)

    def train_unref(self, data_dir, fquery_train, freply_train, fquery_val, freply_val, num_dump, num_dump_val):
        self.unref.train(data_dir, fquery_train, freply_train, fquery_val, freply_val, num_dump, num_dump_val)

    def normalize(self, scores):
        smin = min(scores)
        smax = max(scores)
        diff = smax - smin
        ret = [(s - smin) / diff for s in scores]
        return ret

    def scores(self, data_dir, fquery ,freply, fgenerated, fqvocab, frvocab):
        ref_scores = self.ref.scores(data_dir, freply, fgenerated)
        ref_scores = self.normalize(ref_scores)

        unref_scores = self.unref.scores(data_dir, fquery, fgenerated,
                fqvocab, frvocab)
        unref_socres = self.normalize(unref_scores)

        return [min(a,b) for a,b in zip(ref_scores, unref_scores)]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='hybrid_evaluation.py')
    parser.add_argument('--data_dir', action="store", dest="data_dir")
    parser.add_argument('--qmax_length', action="store", type=int, dest="qmax_length")
    parser.add_argument('--rmax_length', action="store", type=int, dest="rmax_length")
    parser.add_argument('--embed_file', action="store", dest="frword2vec")
    parser.add_argument('--training', action="store_true", default=False, dest="training")
    input_opt = parser.parse_args()
   
    train_dir = input_opt.data_dir + 'trained_models'
    data_dir = input_opt.data_dir
    qmax_length, rmax_length = [input_opt.qmax_length, input_opt.rmax_length]
    frword2vec = input_opt.frword2vec
    training = input_opt.training
     
    if training:
        """train"""        
        fquery_train, freply_train = ['toronto_books_p2_query_train.txt','toronto_books_p2_reply_train.txt']
        fquery_val, freply_val = ['toronto_books_p2_query_valid.txt','toronto_books_p2_reply_valid.txt']
        num_dump = 22
        num_dump_val = 6
        hybrid = Hybrid(data_dir, frword2vec, '%s.embed'%fquery_train, '%s.embed'%freply_train, qmax_length, rmax_length)
        hybrid.train_unref(data_dir, fquery_train, freply_train, fquery_val, freply_val, num_dump, num_dump_val)

    else:
        """test"""
        fquery, freply = ['src-test.txt','tgt-test.txt']
        out_file='pred_t0.001.txt'
        hybrid = Hybrid(data_dir, frword2vec, '%s.embed'%fquery, '%s.embed'%freply, qmax_length, rmax_length)    
        
        scores = hybrid.scores(data_dir, fquery, freply, out_file, '%s.vocab%d'%(fquery, qmax_length),'%s.vocab%d'%(out_file, rmax_length))
        for i, s in enumerate(scores):
            print (i,s)
        print ('avg:%f'%(sum(scores)/len(scores)))
