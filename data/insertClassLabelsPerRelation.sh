#!/bin/bash

if [ "$#" -ne 4 ]; then
	echo "Usage: $0 relation_dir classId mainDir log_dir"
	exit 1
fi

host=`hostname`

# if [[ $host == ip* ]]; then
# 	mainDir='/home/ubuntu/EMNLP/LSTM-KBC'	#for ec2 machines
# else
# 	mainDir='/home/rajarshi/EMNLP/LSTM-KBC'
# fi

relation_dir=$1
classId=$2
mainDir=$3
log_dir=$4

data_set=("train" "dev" "test")

for set in "${data_set[@]}"
do
	echo $set
	dataDir=${relation_dir}/${set}
	for f in `ls $dataDir/*.torch`
	do
		cmd="th $mainDir/data/insertClassLabels.lua -input $f -classLabel $classId"
		echo $cmd 
		$cmd 2>>$log_dir/insertClassLabels.err
	done
done

