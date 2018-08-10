__author__ = 'Sarik'

import os 


def create_files(data_dir, fname, fqoutput, froutput):
    with open(data_dir + fname, 'r') as fin:
        input_lines = fin.readlines()
    with open(fqoutput, 'w') as fout:
        fout.writelines([line for line in input_lines[:-1]])
    with open(froutput, 'w') as fout:
        fout.writelines([line for line in input_lines[1:]])

if __name__ == '__main__':
    data_dir ="toronto_books_in_sentences/"
    fname ='books_large_p2.txt'
    fqoutput = 'data/toronto_books_p2_query.txt'
    froutput = 'data/toronto_books_p2_reply.txt'
    create_files(data_dir, fname, fqoutput, froutput)
    