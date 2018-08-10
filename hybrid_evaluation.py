__author__ = 'liming-vie'

import os
from referenced_metric import Referenced
from unreferenced_metric import Unreferenced

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
    train_dir = 'Ruber/RUBER/data_TrainTestVal/trained_models'
    data_dir = 'Ruber/RUBER/data_TrainTestVal/'
    qmax_length, rmax_length = [200, 200]
    frword2vec = 'small_file_embeddings.txt'
    training = True
    
    if training:
        """train"""        
        fquery_train, freply_train = ['toronto_books_p2_query_train.txt','toronto_books_p2_reply_train.txt']
        fquery_val, freply_val = ['toronto_books_p2_query_valid.txt','toronto_books_p2_reply_valid.txt']
        num_dump = 2
        num_dump_val = 1
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

