from optimize import *
from diagnostics import *
import configs
import argparse
from data import *
import copy
import sys
import json

parser = argparse.ArgumentParser()
parser.add_argument('config')
parser.add_argument('dataset_path')
parser.add_argument('-i', '--initial_params', default=None)
parser.add_argument('-glove', '--glove_vectors', default=None)

args = parser.parse_args()
config = getattr(configs, args.config)
config['dataset_path'] = args.dataset_path

entity_vocab = {}
relation_vocab = {}
out_file = ''
if args.config == 'wordnet_experiment':
	with open('../vocab/wordnet/relation_vocab.txt') as rel_vocab:
		relation_vocab = json.load(rel_vocab)
	with open('../vocab/wordnet/entity_vocab.txt') as ent_vocab:
		entity_vocab = json.load(ent_vocab)
	out_dir = '../negative_examples/wordnet'
	out_files = [out_dir+'/dev_neg.txt', out_dir+'/test_neg.txt']
else:
	with open('../vocab/freebase/relation_vocab.txt') as rel_vocab:
		relation_vocab = json.load(rel_vocab)
	with open('../vocab/freebase/entity_vocab.txt') as ent_vocab:
		entity_vocab = json.load(ent_vocab)
	out_dir = '../negative_examples/freebase'
	out_files = [out_dir+'/dev_neg.txt', out_dir+'/test_neg.txt']

print 'Vocab file read...'

for dset_counter, dset_type in enumerate(['dev', 'test']):
	print dset_type
	dev_mode = False
	if dset_counter == 0:
		dev_mode = True
	dset = parse_dataset(args.dataset_path, dev_mode=dev_mode)
	# used for all evaluations
	neg_gen = NegativeGenerator(dset.full_graph, float('inf'), type_matching_negs=True)
	queries = dset.test
	out_file = out_files[dset_counter]
	with open(out_file, 'w') as out:
		for counter, query in enumerate(util.verboserate(queries)):
			s, r, t = query.s, query.r, query.t
			negatives = neg_gen(query, 't')
			f = lambda x: entity_vocab[x] if x in entity_vocab else entity_vocab['#UNK_TOKEN']
			negatives_int = [str(f(x)) for x in negatives]
			out_str = ','.join(negatives_int)
			out.write(out_str+'\n')
			if counter % 10000 == 0:
				print counter


