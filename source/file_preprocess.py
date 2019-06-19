# preprocess text file
# clean text
# build dictionary
# convert into index to word

import sys
sys.path.append('.')
from preprocessor import *
from collections import Counter
import json

INFILE = sys.argv[1]
OUTFILE = sys.argv[2]
DIC_FILE = sys.argv[3]
VOCAB_SIZE = int(sys.argv[4])

# main
def main(infile, outfile, vocab_size, dic_file):

    dic = preprocess(infile, vocab_size, dic_file)

    index_to_words(infile, outfile, dic)

# process text file to generate dictionary
def preprocess(infile, vocab_size, dic_file):

    prep = NLPPreprocessor()

    fin = open(infile, 'r')

    for text in fin:

        text = prep.remove_contracts(text)
        text = prep.remove_puncts(text)
        words = text.lower().split()
        prep.count_words(words)

    fin.close()

    # build dictionary
    dic = prep.build_vocab(vocab_size)

    # save dictionary
    with open(dic_file, 'w') as fout:

        json.dump(dic, fout)


    return dic

# process text file to generate index to words representation
def index_to_words(infile, outfile, dic):

    prep = NLPPreprocessor()

    fin = open(infile, 'r')

    fout = open(outfile, 'w')

    for text in fin:
        text = prep.remove_contracts(text)
        text = prep.remove_puncts(text)
        words = text.lower().split()
        id_words = prep.word_to_index(words, dic)

        for v in id_words:
            fout.write(str(v))
            fout.write(" ")

    fout.close()
    fin.close()

# call main
main(INFILE, OUTFILE, VOCAB_SIZE, DIC_FILE)



