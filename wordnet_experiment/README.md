#Path Query experiments 

This is the path query experiment in Sec 5.5 of the paper.

##Data
Download the datasets released by the original paper [here](https://worksheets.codalab.org/worksheets/0xfcace41fdeec45f3bc6ddf31107b829f/)

For evaluation, you will also need to have the negative entities for each target entity. You can get that by running data/get_negative_examples.py. (You have to run it in the original code repository.). I am unable to share the negative entities because of the data limit of github repositories.

##Running the model
Checkout model/run.lua. It defines all the hyper-params and sets other required variables. If you want to train the model on a CPU instead of a GPU just set ```useCuda = False```. To start training the model
```
th run.lua
```



