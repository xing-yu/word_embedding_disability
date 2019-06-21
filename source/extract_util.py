# extract text from json files

import json
import sys

INFILES = [sys.argv[1], sys.argv[2]]
TYPES = [sys.argv[3], sys.argv[4]]
OUTFILE = sys.argv[5]

# main
def main(infiles, types, outfile):

	for i in range(len(infiles)):

		extract_text(infiles[i], outfile, types[i])

# extract text from comments and posts json files
def extract_text(infile, outfile, input_type):

	fin = open(infile, 'r')

	fout = open(outfile, 'a')

	for line in fin:

		data = json.loads(line)

		try:

			# if input is comment file
			if input_type == "c":

				text = data["body"]

			else:

				text = data["selftext"]

		except Exception as e:
			print(e)
			pass

		if text:

			fout.write(text + "\n")

	fin.close()
	fout.close()

# call main
main(INFILES, TYPES, OUTFILE)




