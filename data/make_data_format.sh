#!/bin/bash
host=`hostname`

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 data_dir out_dir" >&2
  exit 1
fi


mainDir='/home/rajarshi/ChainsofReasoning'
data_dir=$1
out_dir=$2
#data_dir='/iesl/canvas/rajarshi/release_with_entities_akbc_linearized_remove_duplicates/'
#out_dir='/iesl/local/rajarshi/data_full_max_length_8_wout_duplicates/'
preprocessingDir=${mainDir}/'data'
only_relation=0 #this means that the input release_directory has only relations.
qsub_log_dir=$preprocessingDir/qsub_log
max_path_length=8
num_entity_types=8
get_only_relations=0 #usually 0, if 1 then get only relations

echo "max_path_length "$max_path_length
echo "num_entity_types "$num_entity_types
echo "get_only_relations" $get_only_relations

mkdir -p $out_dir
mkdir -p $qsub_log_dir
for r in `ls ${data_dir} | grep ^_`
do
	relation_dir=$data_dir/$r
	out_dir_r=$out_dir/$r
	name=$r
	echo "Starting for "$relation_dir.
	# qsub -o $qsub_log_dir/$r.log -e $qsub_log_dir/$r.err -m e -M rajdas14.2007@gmail.com -N $name  -S /bin/bash $preprocessingDir/make_data_format_single_relation.sh $relation_dir $out_dir_r $r $num_entity_types $only_relation &
	cmd="/bin/bash $preprocessingDir/make_data_format_single_relation.sh $relation_dir $out_dir_r $r $only_relation $max_path_length $num_entity_types $get_only_relations"
	echo $cmd
	$cmd 2>$qsub_log_dir/$r.err &
done