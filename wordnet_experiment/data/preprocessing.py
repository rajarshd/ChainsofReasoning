import json
from collections import defaultdict
import argparse
import os

#Usage: python preprocessing.py -m 1 -i /home/rajarshi/canvas/traversing_knowledge_graphs/data/path_datasets/ -o /home/rajarshi/canvas/traversing_knowledge_graphs/data_formatted

parser = argparse.ArgumentParser()
parser.add_argument('-m','--make_vocab', default=0, type=int)
parser.add_argument('-i','--input_dir', required=True)
parser.add_argument('-o','--output_dir', required=True)

args = parser.parse_args()
make_vocab = (args.make_vocab == 1)
parent_dir = args.input_dir
OUTPUT_DIR = args.output_dir

# parent_dir = "/home/rajarshi/canvas/traversing_knowledge_graphs/data/path_datasets/"
# OUTPUT_DIR = "/home/rajarshi/canvas/traversing_knowledge_graphs/data_formatted/"

data_dirs = ['/wordnet_socher_paths_test', '/freebase_socher_paths_test']
output_dirs = ['/wordnet','/freebase']


PAD_TOKEN='#PAD_TOKEN'
UNK_TOKEN='#UNK_TOKEN'

if make_vocab:
	for count, input_dir in enumerate(data_dirs):
		print 'Creating vocab for {}'.format(input_dir)
		input_dir = parent_dir + input_dir
		train_file = input_dir+'/train'
		entity_counts = defaultdict(int)
		entity_vocab = {}
		relation_counts = defaultdict(int)
		relation_vocab = {}
		entity_counter = 0
		relation_counter = 0
		print('Reading train file...')
		with open(train_file) as train:
			for line in train:
				line = line.strip()
				e1, path, e2 = line.split('\t')
				entity_counts[e1] = entity_counts[e1] + 1
				entity_counts[e2] = entity_counts[e2] + 1
				relations = path.split(',')
				for relation in relations:
					relation = relation.strip()
					relation_counts[relation] = relation_counts[relation] + 1

		sorted_entity_counts = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)
		entity_vocab = {k:counter for counter, (k,_) in enumerate(sorted_entity_counts)}

		sorted_relation_counts = sorted(relation_counts.items(), key=lambda x:x[1], reverse=True)
		relation_vocab = {k:counter for counter, (k,_) in enumerate(sorted_relation_counts)}

		relation_vocab[UNK_TOKEN] = len(relation_vocab)
		relation_vocab[PAD_TOKEN] = -1 #len(relation_vocab) #the reason I am setting it as -1 is because when saving the data in lua format, we will add 1 and then pad token would be 0 which makes life easier with LookupTableMaskZero
		entity_vocab[UNK_TOKEN] = len(entity_vocab)

		print 'There are {} entity tokens and {} relation tokens in the train set'.format(len(entity_vocab), len(relation_vocab))

		relation_vocab_file = OUTPUT_DIR+output_dirs[count]+"/vocab/relation_vocab.txt"
		entity_vocab_file = OUTPUT_DIR+output_dirs[count]+"/vocab/entity_vocab.txt"

		print('Writing relation vocab...')
		with open(relation_vocab_file,'w') as out:
			json.dump(relation_vocab, out)

		print('Writing entity vocab...')
		with open(entity_vocab_file,'w') as out:
			json.dump(entity_vocab, out)

input_files = ['/train','/dev','/test']
output_files = ['/train.int','/dev.int','/test.int','/dev_kbc.int','/test_kbc.int']
neg_files = ['/dev_neg.int','/test_neg.int']
out_neg_files = ['/dev_kbc_neg.int','/test_kbc_neg.int']



#Now convert the files to int
for dir_count, input_dir in enumerate(data_dirs):
	# these two lists will contain the line counter so that we can get the negative entities for each fact for the KBC experiment 
	dev_kbc_line_counter = []
	test_kbc_line_counter = []
	print 'Creating data for {}'.format(input_dir)
	input_dir = parent_dir + input_dir
	output_dir = OUTPUT_DIR + output_dirs[dir_count]
	#delete the /dev_kbc.int','/test_kbc.int file
	if os.path.exists(output_dir + output_files[3]):
		os.remove(output_dir + output_files[3])
	if os.path.exists(output_dir + output_files[4]):
		os.remove(output_dir + output_files[4])
	for file_count, input_file in enumerate(input_files):
		input_file = input_dir + input_file
		output_file = output_dir + output_files[file_count]
		entity_vocab_file = output_dir+'/vocab/entity_vocab.txt'
		relation_vocab_file = output_dir+'/vocab/relation_vocab.txt'
		entity_vocab, relation_vocab = {}, {}
		with open(entity_vocab_file) as input:
			entity_vocab = json.load(input)
		with open(relation_vocab_file) as input:
			relation_vocab = json.load(input)
		#go through the file to get the max length of the path
		MAX_LENGTH = -1
		with open(input_file) as input:
			for line in input:
				line = line.strip()
				output_line = ''
				e1, path, e2 = line.split('\t')
				length = len(path.split(','))
				if length > MAX_LENGTH:
					MAX_LENGTH = length
		print 'Max length is {}'.format(MAX_LENGTH)
		with open(output_file, 'w') as out:
			with open(input_file) as input:
				for line_counter, line in enumerate(input):
					line = line.strip()
					output_line = ''
					e1, path, e2 = line.split('\t')
					relations = path.split(',')
					int_path = map(lambda x: str(relation_vocab[x]) if x in relation_vocab else str(relation_vocab[UNK_TOKEN]), relations)
					pad_length = MAX_LENGTH - len(relations)
					pad_tokens = [str(relation_vocab[PAD_TOKEN]) for i in xrange(pad_length)]
					if len(relations) == 1: #if query length 1 save it to a different file
						if file_count == 1: 
							f = lambda x: entity_vocab[x] if x in entity_vocab else entity_vocab[UNK_TOKEN]
							output_line = '\t'.join([str(f(e1)), ','.join(int_path),str(f(e2))])
							with open(output_dir + output_files[3],'a') as out_dev_kbc:
								out_dev_kbc.write(output_line+'\n')
							dev_kbc_line_counter.append(line_counter)
						if file_count == 2: 
							f = lambda x: entity_vocab[x] if x in entity_vocab else entity_vocab[UNK_TOKEN]
							output_line = '\t'.join([str(f(e1)), ','.join(int_path),str(f(e2))])
							with open(output_dir + output_files[4],'a') as out_test_kbc:
								out_test_kbc.write(output_line+'\n')
							test_kbc_line_counter.append(line_counter)
					int_path = pad_tokens + int_path
					f = lambda x: entity_vocab[x] if x in entity_vocab else entity_vocab[UNK_TOKEN]
					output_line = '\t'.join([str(f(e1)), ','.join(int_path),str(f(e2))])
					out.write(output_line+'\n')
	print('Writing the negative entities for KBC experiment (single hops)')
	for neg_file_counter, neg_file in enumerate(neg_files):
		neg_file = output_dir + neg_file
		out_neg_file = output_dir + out_neg_files[neg_file_counter]
		line_counter_list = []
		if neg_file_counter == 0:
			line_counter_list = dev_kbc_line_counter
		else:
			line_counter_list = test_kbc_line_counter
		with open(neg_file) as neg_entities_input:
			print(neg_file)
			with open(out_neg_file, 'w') as neg_entities_output:
				print(out_neg_file)
				pointer = 0
				for line_counter, line in enumerate(neg_entities_input):
					line = line.strip()
					if line_counter == line_counter_list[pointer]:
						neg_entities_output.write(line+'\n') #write the corresponding line from neg_entities_input to output
						pointer = pointer + 1
						if pointer == len(line_counter_list):
							break




