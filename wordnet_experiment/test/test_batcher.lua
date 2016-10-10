package.path = package.path ..';../data/?.lua'

require 'Batcher'

local input_file = '/iesl/canvas/rajarshi/traversing_knowledge_graphs/data_formatted/wordnet/train.torch'
local batch_size = 100
local entity_vocab_size = 38195
local shuffle = true

local batcher = Batcher(input_file, batch_size, entity_vocab_size, shuffle)

print('Test 1: See if the number of data points are returned correctly.')
local counter = 0
while true do
	local t = batcher:get_batch()
	if t == nil then break end
	counter = counter + t[1]:size(1)
end
assert(counter == batcher:get_data_size())

print('Test 1 passed!')

print('Test 2: Check if reset works')
batcher:reset()
counter = 0
while true do
	local t = batcher:get_batch()
	if t == nil then break end
	counter = counter + t[1]:size(1)
end
assert(counter == batcher:get_data_size())
print('Test 2 passed!')


print('Test 3: Check if returning (precomputed) negative samples for entity works')
input_file = '/iesl/canvas/rajarshi/traversing_knowledge_graphs/data_formatted/wordnet/dev.torch'
local neg_input_file = '/iesl/canvas/rajarshi/traversing_knowledge_graphs/data_formatted/wordnet/dev_neg.torch'
shuffle = false
local return_neg = true
batcher = Batcher(input_file, batch_size, entity_vocab_size, shuffle, return_neg, neg_input_file)
while true do
	local t = batcher:get_batch()
	if t == nil then break end
	assert(#t == 6)
	assert(t[5]:size(2) == entity_vocab_size)
	assert(t[5]:size(1) <= batch_size)
end
print('Test 3 passed!')