import json
from collections import defaultdict
import argparse
import os


neg_input_file_test ='/home/rajarshi/canvas/traversing_knowledge_graphs/negative_examples/wordnet/test_neg.int' 
neg_input_file_dev = '/home/rajarshi/canvas/traversing_knowledge_graphs/negative_examples/wordnet/dev_neg.int'
neg_output_file_test ='/home/rajarshi/canvas/traversing_knowledge_graphs/negative_examples/wordnet/test_neg_uniq.int' 
neg_output_file_dev = '/home/rajarshi/canvas/traversing_knowledge_graphs/negative_examples/wordnet/dev_neg_uniq.int'

neg_input_files = [neg_input_file_dev, neg_input_file_test]
neg_output_files = [neg_output_file_dev, neg_output_file_test]

for file_counter, neg_file in enumerate(neg_input_files):
	with open(neg_output_files[file_counter],'w') as out:
		with open(neg_file) as input:
			print('Processing {}'.format(neg_file))
			for line in input:
				line = line.strip()
				splits = line.split(',')
				split_set = set(splits)
				output_line = ','.join(split_set)
				out.write(output_line+'\n')
	print('Processed {}'.format(neg_file))






