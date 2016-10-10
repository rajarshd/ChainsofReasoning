-- Test cases for the network
package.path = package.path ..';../model/?.lua'
package.path = package.path ..';../data/?.lua'
require 'PathQA'
require 'Batcher'
require 'cunn'


args = {
input_dim = 250,
relation_vocab_size = 24,
entity_vocab_size = 38195,
output_dim = 250
}
local p_qa = PathQA(args)
local net = p_qa:get_network()
net = net:cuda()

local params, gradParams = net:parameters()



--Test 1: See if the lookuptables are actually shared.
print('Test 1: See if the lookuptables are actually shared.')
local e1_lookuptable = params[1][1]
local e2_lookuptable = params[7][1]
local e_neg_lookuptable = params[8][1]

for i=1, e1_lookuptable:size(1) do
	assert(e1_lookuptable[i] == e2_lookuptable[i])
	assert(e2_lookuptable[i] == e_neg_lookuptable[i])
end

--Now set one to zero and check if all are set too
e1_lookuptable:zero()
for i=1, e1_lookuptable:size(1) do
	assert(e1_lookuptable[i] == e2_lookuptable[i])
	assert(e2_lookuptable[i] == e_neg_lookuptable[i])
end

e2_lookuptable:add(1)
for i=1, e1_lookuptable:size(1) do
	assert(e1_lookuptable[i] == e2_lookuptable[i])
	assert(e2_lookuptable[i] == e_neg_lookuptable[i])
end

e_neg_lookuptable:uniform(-0.1, 0.1)
for i=1, e1_lookuptable:size(1) do
	assert(e1_lookuptable[i] == e2_lookuptable[i])
	assert(e2_lookuptable[i] == e_neg_lookuptable[i])
end

print('Test passed!')

print('Test 2: the forward method -- check shape')
local input_file = '/iesl/canvas/rajarshi/traversing_knowledge_graphs/data_formatted/wordnet/train.torch'
local batch_size = 100
local entity_vocab_size = 38195
local shuffle = true
local batcher = Batcher(input_file, batch_size, entity_vocab_size, shuffle)

local batch_data = batcher:get_batch()

local out = net(batch_data)
assert(#out == 2)
print('Test passed!')