__author__ = 'liming-vie'

import os
import _pickle as cPickle
import dill
import numpy as np
from tensorflow.contrib import learn
import math

def tokenizer(iterator):
    for value in iterator:
        yield value.split()

def load_file(data_dir, fname):
    fname = os.path.join(data_dir, fname)
    print ('Loading file %s'%(fname))
    lines = open(fname).readlines()
    return [line.rstrip() for line in lines]

def process_train_file(data_dir, fname, max_length, size_dump, min_frequency=10):
    """
    Make vocabulary and transform into id files

    Return:
        vocab_file_name
        vocab_dict: map vocab to id
        vocab_size
    """
    fvocab = '%s.vocab%d'%(fname, max_length)
    foutput = os.path.join(data_dir, fvocab)
    if os.path.exists(foutput):
        print ('Loading vocab from file %s'%foutput)
        vocab = load_vocab(data_dir, fvocab)
        print('number of vocabs is ' +str(len(vocab)))
        return fvocab, vocab, len(vocab)

    vocab_processor = learn.preprocessing.VocabularyProcessor(max_length,
            tokenizer_fn = tokenizer, min_frequency=min_frequency)
    x_text = load_file(data_dir, fname)
    print ('Vocabulary transforming')
    # will pad 0 for length < max_length
    ids = list( vocab_processor.fit_transform(x_text))
    num_dump = math.ceil(len(ids)/size_dump)
    
    low_b = 0
    high_b = size_dump
    for ind in range(0, num_dump):
        fid = os.path.join(data_dir, fname + str(ind) +'.id%d'%max_length)
        print('Saving %s ids file %i in %s'%(fname, ind, fid))
        if ind == num_dump-1:	
                cPickle.dump(ids[low_b:], open(fid, 'wb'), protocol=2)
                print(str(low_b) + " " + str(len(ids)))
        else:
                cPickle.dump(ids[low_b:high_b], open(fid, 'wb'), protocol=2)
                print(str(low_b) + " " + str(high_b))
        low_b = high_b
        high_b = low_b + size_dump
    
   
    '''ids_copy = []
    data = load_data(data_dir, fname, num_dump , max_length)
    first = True
    for m in data:
        ids_copy.append(m[1])
    print('copy of ids from several dumps has length of  ' + str(len(ids_copy)))
    print('check if two dumped data are the same ' )
    print('type of ids_copy is ' + str(type(ids_copy)))
    print('type of ids is ' + str(type(ids)))
    #print(ids[100])
    #print(ids_copy[100])
    t_ids = [tuple(y) for y in ids]
    t_ids_copy = [tuple(y) for y in ids_copy]
    if set(t_ids) == set(t_ids_copy):
       print('both are the same')'''

    print ('Saving vocab file in %s'%foutput)
    size = len(vocab_processor.vocabulary_)
    vocab_str = [vocab_processor.vocabulary_.reverse(i) for i in range(size)]
    with open(foutput, 'w') as fout:
        fout.write('\n'.join(vocab_str))

    vocab = load_vocab(data_dir, fvocab)
    print('number of vocabs is ' +str(len(vocab)))
    print('first element is the vocab file is ' + str(list(vocab.keys())[0]))
    for m in ids[0]:
        for vo,ind in vocab.items():
            if ind == m:
                print('word ' + str(m) + " is " + vo)
    return fvocab, vocab, len(vocab)

def process_valid_file(data_dir, fname, fvocab_train, max_length, size_dump, min_frequency=10):
    """
    Make vocabulary and transform into id files

    Return:
        valid_vocab_file_name
        valid_vocab_dict: map vocab to id
        valid_vocab_size
    """ 
    vocab = load_vocab(data_dir, '%s.vocab%d'%(fvocab_train, query_max_length))    
    print('training data has ' + str(len(vocab)) + ' vocabularies.' )
    #for vo,ind  in vocab.items():
        #if ind == 69 or ind==37 or ind== 3845:
            #print("word" +str(ind)+ " is " + vo)

    x_valid = load_file(data_dir, fname)
    #print('first sentence in valid file is ' + x_valid[0])
    #print('100th sentence in valid file is ' + x_valid[99])
 
    ids = []
    #max_id =0
    #index = 0
    for line in x_valid:
        ids_part= []
        words_query = line.split()
        for word in words_query:
            #if index == 0 or index == 1:
                #print('word ' + word + " index " + str(vocab.get(word,0)))
            ids_part.append(vocab.get(word,0)) 
        if len(ids_part) < max_length:
            ids_part = np.pad(ids_part, (0, max_length-len(ids_part)), 'constant')
        ids.append(ids_part)
        #if index ==0:
            #print("********")
        #index +=1
    #print(ids[0])
    
    for m in ids:
        for n in m:
            if n > max_id:
                max_id = n

    #print('max id in valid file .id is ' + str(max_id))
    num_dump = math.ceil(len(ids)/size_dump)
  
    low_b = 0
    high_b = size_dump
    for ind in range(0, num_dump):
        fid = os.path.join(data_dir, fname + str(ind) +'.id%d'%max_length)
        print('Saving %s ids file %i in %s'%(fname, ind, fid))
        if ind == num_dump-1:	
                cPickle.dump(ids[low_b:], open(fid, 'wb'), protocol=2)
                print(str(low_b) + " " + str(len(ids)))
        else:
                cPickle.dump(ids[low_b:high_b], open(fid, 'wb'), protocol=2)
                print(str(low_b) + " " + str(high_b))
        low_b = high_b
        high_b = low_b + size_dump    
    return 

def load_data(data_dir, fname, num_dump, max_length):
    """
    Read id file data

    Return:
        data list: [[length, [token_ids]]]
    """
    data=[]
    for ind in range(0, num_dump):
       fname1 = os.path.join(data_dir, "%s%i.id%d"%(fname, ind, max_length))
       ids = cPickle.load(open(fname1, 'rb')) 
       print('file ' + str(ind) + " has " +str(len(ids)) + " ids")
       for vec in ids:
           length = len(vec)
           if vec[-1] == 0:
                length = list(vec).index(0)
           data.append([length, vec])        
    return data

def load_vocab(data_dir, fvocab):
    """
    Load vocab
    """
    fvocab = os.path.join(data_dir, fvocab)
    print ('Loading vocab from %s'%fvocab)
    vocab={}
    with open(fvocab) as fin:
        for i, s in enumerate(fin):
            vocab[s.rstrip()] = i
    return vocab

def transform_to_id(vocab, sentence, max_length):
    """
    Transform a sentence into id vector using vocab dict
    Return:
        length, ids
    """
    words = sentence.split()
    ret = [vocab.get(word, 0) for word in words]  
    l = len(ret)
    l = max_length if l > max_length else l
    if l < max_length:
        ret.extend([0 for _ in range(max_length - l)])
    return l, ret[:max_length]

def make_embedding_matrix(data_dir, fname, word2vec, vec_dim, fvocab):
    foutput = os.path.join(data_dir, '%s.embed'%fname)
    if os.path.exists(foutput):
        print ('Loading embedding matrix from %s'%foutput)
        m = cPickle.load(open(foutput, 'rb'))
        print(len(m))
        return cPickle.load(open(foutput, 'rb'))
    vocab_str = load_file(data_dir, fvocab)
    print ('Saving embedding matrix in %s'%foutput)
    matrix=[]
    for vocab in vocab_str:
        vec = word2vec[vocab] if vocab in word2vec \
                else [0.0 for _ in range(vec_dim)]
        matrix.append(vec)
    cPickle.dump(matrix, open(foutput, 'wb'), protocol=2)
    print('number of words in embedding matrix is ' + str(len(matrix)))
    load = cPickle.load(open(foutput,'rb'))
    return matrix

def load_word2vec(data_dir, fword2vec):
    """
    Return:
        word2vec dict
        vector dimension
        dict size
    """
    fword2vec = os.path.join(data_dir, fword2vec)
    print ('Loading word2vec dict from %s'%fword2vec)
    vecs = {}
    vec_dim=0
    i = 1
    with open(fword2vec, errors='ignore') as fin:
        size, vec_dim = map(int, fin.readline().split())
        for line in fin:
            if i % 1000000 == 0:
                print(i)
            i +=1
            ps = line.rstrip().split()
            vecs[ps[0]] = list(map(float, ps[1:]))
    return vecs, vec_dim, size

if __name__ == '__main__':
    data_dir = 'Ruber/RUBER/data_TrainTestVal/'
    query_max_length, reply_max_length = [200, 200]
    fquery_train, freply_train = ['toronto_books_p2_query_train.txt','toronto_books_p2_reply_train.txt']
    fquery_valid, freply_valid = ['toronto_books_p2_query_valid.txt','toronto_books_p2_reply_valid.txt']
    #fquery_test, freply_test = ['src-test.txt','pred_t0.001.txt']
    fword2vec = 'file_embeddings.txt'
    size_dump = 1000000
   
    process_train_file(data_dir, fquery_train, query_max_length, size_dump) #creates/loads the vocab file of query (src)
    process_train_file(data_dir, freply_train, reply_max_length, size_dump) #creates/loads the vocab file of reply (tgt)
    process_valid_file(data_dir, fquery_valid, fquery_train, query_max_length, size_dump) #just need to create .ids file based on the vocab in train_file
    process_valid_file(data_dir, freply_valid, freply_train, reply_max_length, size_dump)        
     
    fqvocab = '%s.vocab%d'%(fquery_train, query_max_length)
    frvocab = '%s.vocab%d'%(freply_train, reply_max_length)

    word2vec, vec_dim, _ = load_word2vec(data_dir, fword2vec) # load word embeddings of whole 3000000 words into word2vec
    make_embedding_matrix(data_dir, fquery_train, word2vec, vec_dim, fqvocab) # make embedding matrix just for words in fqvocab
    make_embedding_matrix(data_dir, freply_train, word2vec, vec_dim, frvocab)
   
    '''data=[]
    for ind in range(0, 1):
       fname1 = os.path.join(data_dir, "%s%i.id%d"%(freply_valid, ind, reply_max_length))
       ids = cPickle.load(open(fname1, 'rb'))
       print('file ' + str(ind) + " has " +str(len(ids)) + " ids")
       for vec in ids:
           length = len(vec)
           if vec[-1] == 0:
                length = list(vec).index(0)
           data.append([length, vec])
    print(data[0][1])
    print(data[99][1])
    vocab = load_vocab(data_dir, frvocab)
    for m in data[0][1]:
        for vo,ind  in vocab.items():
            if ind == m:
                print('word ' + vo)

    for m in data[99][1]:
        for vo,ind  in vocab.items():
            if ind == m:
                print('word ' + str(m) + ' is '+ vo )
    '''
    pass
