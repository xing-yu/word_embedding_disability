# preprocess text file
# clean text
# build dictionary
# convert into index to word

#python3 ./source/file_preprocess.py /home/yu64/Documents/IdentificationProjectNeo/data/posts_comments.txt ./result/idx2words.txt ./dic/dic.json 400000000

#python3 ./source/file_preprocess.py /Users/Xing/jupyter_notebook_server/word2vec/data/comments.txt ./result/idx2words.txt ./dic/dic.json 400000000

import sys
sys.path.append('.')
from preprocessor import *
from collections import Counter
import json
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))

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
    ps = PorterStemmer()

    fin = open(infile, 'r')

    for text in fin:
        
        if text == "[removed]": continue

        text = prep.remove_contracts(text)
        text = prep.remove_puncts(text)
        words = text.lower().split()
        words = [ps.stem(w) for w in words if w not in stop_words and len(w) < 20 and w.isdigit() == False]

        prep.count_words(words, vocab_size)

    fin.close()

    # build dictionary

    dic = prep.build_vocab(vocab_size)

    # save dictionary
    with open(dic_file, 'w') as fout:

        json.dump(dic, fout)

    print("The size of dictionary is " + str(len(dic)))

    return dic

# process text file to generate index to words representation
def index_to_words(infile, outfile, dic):

    prep = NLPPreprocessor()
    ps = PorterStemmer()

    fin = open(infile, 'r')

    fout = open(outfile, 'w')

    for text in fin:

        if text == "[removed]": continue

        text = prep.remove_contracts(text)
        text = prep.remove_puncts(text)
        words = text.lower().split()
        words = [ps.stem(w) for w in words if w not in stop_words and len(w) < 20 and w.isdigit() == False]
        id_words = prep.word_to_index(words, dic)

        for v in id_words:
            fout.write(str(v))
            fout.write(" ")

    fout.close()
    fin.close()

# call main
main(INFILE, OUTFILE, VOCAB_SIZE, DIC_FILE)



