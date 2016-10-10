#!/bin/bash
#sh data_processing.sh /home/rajarshi/canvas/traversing_knowledge_graphs/data/path_datasets/ /home/rajarshi/canvas/traversing_knowledge_graphs/data_formatted/

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 input_dir output_dir" >&2
  echo "input_dir is where the parent directory of wordnet and freebase datasets"
  exit 1
fi

# input_dir='/home/rajarshi/canvas/traversing_knowledge_graphs/data/path_datasets/'
# output_dir='/home/rajarshi/canvas/traversing_knowledge_graphs/data_formatted/'

input_dir=$1
output_dir=$2
script_dir='/home/rajarshi/eacl17_experiments_wordnet/code/data'
log_dir=$script_dir/logs

mkdir -p $output_dir/wordnet/vocab
mkdir -p $output_dir/freebase/vocab
mkdir -p $log_dir


py_cmd="python $script_dir/preprocessing.py -m 0 -i $input_dir -o $output_dir"
$py_cmd 2>$log_dir/log.err

th_cmd="th create_tensors.lua -input_dir $output_dir -output_dir $output_dir"
$th_cmd 2>>$log_dir/log.err