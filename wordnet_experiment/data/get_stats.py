#get stats like number of relations, entity pairs etc etc.
from collections import defaultdict

input_dir = "/home/rajarshi/canvas/traversing_knowledge_graphs/data/wordnet"

input_files = ['train', 'dev', 'test']

for input_file in input_files:
	print input_file
	input_file = input_dir+'/'+input_file
	entity_set = set()
	relation_set = set()
	entity_map = defaultdict(int)
	with open(input_file) as input:
		for line_counter, line in enumerate(input):
			line = line.strip()
			ent1, path, ent2 = line.split('\t')
			entity_set.add(ent1)
			entity_set.add(ent2)
			
			relations = path.split(',')
			for relation in relations:
				relation_set.add(relation)

	print 'Num unique relations {}'.format(len(relation_set))
	print 'Num unique entities {}'.format(len(entity_set))