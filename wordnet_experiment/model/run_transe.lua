package.path = package.path ..';../data/?.lua'
require 'PathQATransE'
require 'Batcher'
require 'BPRLoss'
require 'optim'
print('USING GPU')
require 'cutorch'
require('cunn')

local gpuid = 0
cutorch.setDevice(gpuid + 1)
local model_save_dir = '/iesl/canvas/rajarshi/path_qa_experiment/'
--params
args = {
input_dim = 100,
relation_vocab_size = 24,
entity_vocab_size = 38195,
output_dim = 100,
useCuda = true,
model_save_dir = model_save_dir
}

local p_qa = PathQATransE(args)
--initialize
p_qa:initialize_net()


local batch_size = 1000
local test_batch_size = 100
local train_file = '/iesl/canvas/rajarshi/traversing_knowledge_graphs/data_formatted/wordnet/train.torch'
local dev_file = '/iesl/canvas/rajarshi/traversing_knowledge_graphs/data_formatted/wordnet/dev.torch'
local test_file = '/iesl/canvas/rajarshi/traversing_knowledge_graphs/data_formatted/wordnet/test.torch'
local neg_input_dev = '/iesl/canvas/rajarshi/traversing_knowledge_graphs/data_formatted/wordnet/dev_neg.torch'
local neg_input_test = '/iesl/canvas/rajarshi/traversing_knowledge_graphs/data_formatted/wordnet/test_neg.torch'
local shuffle = true
local useCuda = true
local train_batcher = Batcher(train_file, batch_size, args.entity_vocab_size, shuffle)
local dev_batcher = Batcher(dev_file, test_batch_size, args.entity_vocab_size, false,true, neg_input_dev)
local test_batcher = Batcher(test_file, test_batch_size, args.entity_vocab_size, false,true, neg_input_test)
local learning_rate = 1e-3
local num_epochs = 7
local grad_clip_norm = 2
local s_criterion = 'bpr'
local margin = 3
local criterion = nil
if s_criterion == 'margin' then criterion = nn.MarginRankingCriterion(margin) else criterion = nn.BPRLoss() end
print(criterion)
local beta1 = 0.9
local beta2 = 0.999
local epsilon = 1e-8
local optim_method = optim.adam


local train_params = {
	train_batcher = train_batcher,
	dev_batcher = dev_batcher,
	test_batcher = test_batcher,
	learning_rate = learning_rate,
	num_epochs = num_epochs,
	grad_clip_norm = grad_clip_norm,
	criterion = criterion,
	beta1 = beta1,
	beta2 = beta2,
	epsilon = epsilon,
	optim_method = optim_method
}

-- print(train_params)

print('Starting to train!')
p_qa:train(train_params)
