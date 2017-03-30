import sys,os
import json
import re
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('-i','--input_dir', required=True)
parser.add_argument('-e','--epoch_file', required=True)
args = parser.parse_args()
input_dir= args.input_dir
epoch_file = args.epoch_file
# model_name = args.model_name

models = []
with open(epoch_file) as input:
	for line in input:
		models.append(line.strip())

data_sets=['test','dev']
for model_name in models:
	for data_set in data_sets:
		input_score_file = input_dir+'/'+data_set+'.scores.'+model_name
		key_file = input_dir+'/'+data_set+'.key.'+model_name
		response_file = input_dir+'/'+data_set+'.response.'+model_name
		if not os.path.isfile(input_score_file):
			continue
		with open(response_file,'w') as response_out:
			with open(key_file,'w') as key_out:
				with open(input_score_file) as input_file:
					for each_line in input_file:
						split = each_line.strip().split('\t')
						predictor_name = split[0]
						entity_pair_id = split[1]
						score = split[2]
						label = split[3]
						if label == '1':
							key_out.write(predictor_name+' 0 '+predictor_name+'_'+entity_pair_id+' 1\n')
						response_out.write(predictor_name+'\t0\t'+predictor_name+'_'+entity_pair_id+'\t0\t'+score+'\t'+'random\n')