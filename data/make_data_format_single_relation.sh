#!/bin/bash

host=`hostname`
if [[ $host == ip* ]]; then
	mainDir='/home/ubuntu/ChainsofReasoning'	#for ec2 machines
	python_exec='/home/ubuntu/anaconda/bin/python'
	ec2_instance=1
else
	mainDir='/home/rajarshi/ChainsofReasoning'
	python_exec='/share/apps/python/bin/python'
	ec2_instance=0
fi
mainDir='/home/rajarshi/ChainsofReasoning'
preprocessingDir=${mainDir}/'data'

relation_dir=$1
out_dir=$2
relation_name=$3
only_relation=$4
max_path_length=$5
num_entity_types=$6
get_only_relations=$7

python $preprocessingDir/make_data_format.py -i $relation_dir -d $out_dir -o $only_relation -e $ec2_instance -m $max_path_length -t $num_entity_types -g $get_only_relations #g is get inly relatoions
data_set=("train" "dev" "test")
int2torch="th ${preprocessingDir}/int2torch.lua"
for set in "${data_set[@]}"
do
	echo $set
	dataDir=${out_dir}/${set}
	dataset=$set
	if [ -f ${out_dir}/${dataset}.list ]; then
		rm ${out_dir}/${dataset}.list #the downstream training code reads in this list of filenames of the data files, split by length
	fi
	echo "converting $dataset to torch files"	
	for ff in $dataDir/*.int 
	do
		out=`echo $ff | sed 's|.int$||'`.torch
		$int2torch -input $ff -output $out -tokenLabels 0 -tokenFeatures 1 -addOne 1 #convert to torch format
		if [ $? -ne 0 ]
		then
			echo 'int2torch failed!'1>&2
			echo 'Failed for relation '$f 1>&2
			continue #continue to the next one
		fi
		echo $out >>  ${out_dir}/${dataset}.list		
		echo ${out}
	done
done