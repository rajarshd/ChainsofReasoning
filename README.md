# ChainsofReasoning

Code for paper [Chains of Reasoning over Entities, Relations, and Text using
Recurrent Neural Networks](http://arxiv.org/pdf/1607.01426v1.pdf)

## Instructions for running the code

### Data
Get the data from [here](http://iesl.cs.umass.edu/downloads/akbc16/). (Note: This might change soon, as I will release an updated version of the dataset)

To get the correct format to run the models, 
```shell
cd data
sh make_data_format.sh <path_to_input_data> <output_dir>
```
For example you can run,
```shell
cd data
sh make_data_format.sh examples/data_small_input examples/data_small_output
```

