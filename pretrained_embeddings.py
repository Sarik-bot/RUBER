__author__ = 'Sarik'
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
import os
import chardet

def get_embed(data_dir, fname):
    """
    Args:
        data_dir: directory of input file
        fname: word2vec pretrained embeddings file name
    """
    fname = os.path.join(data_dir, fname)
    model = gensim.models.KeyedVectors.load_word2vec_format(fname, binary=True, unicode_errors = "ignore")
    model.save_word2vec_format(data_dir + 'file_embeddings_txt.txt', binary=False)
    print('file is saved!')
    return model
    
if __name__ == '__main__':
    data_dir = 'data/'
    #fname = 'GoogleNews-vectors-negative300.bin'
    #model = get_embed(data_dir, fname)
    #result = model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
    #print(result)
    #vectors = [model[w] for w in sentence]
    
    
    
    #f= open(data_dir+'file_embeddings_txt.txt', encoding="utf8")
    #i =0
    #for line in f:
        #print(str(i) + " "  + line.split()[0])
        #i+=1
        
    