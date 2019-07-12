# merge embeddings and words into cornell format
# token vector
# I 0.6 0.8 0.9

import json
import sys

dictionary_file = sys.argv[1]
embeddings_file = sys.argv[2]
output_file = sys.argv[3]

def merge(dictionary_file, embeddings_file, output_file):

	mapping = json.load(open(dictionary_file, 'r'))

	rev_mapping = {v : k for k, v in mapping.items()}

	idx = 0

	fout = open(output_file, 'w')

	with open(embeddings_file, 'r') as f:

		for line in f:

			fout.write(rev_mapping[idx])
			fout.write(' ')
			fout.write(line)
			fout.write('\n')

			idx += 1

	fout.close() 

merge(dictionary_file, embeddings_file, output_file)


