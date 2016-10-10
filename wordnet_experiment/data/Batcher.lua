local Batcher = torch.class('Batcher')

require 'nn'


function Batcher:__init(input_file, batch_size, entity_vocab_size, shuffle, return_neg, neg_input_file)
	-- body
	self.input_file = input_file
	self.batch_size = batch_size
	self.entity_vocab_size = entity_vocab_size
	self.do_shuffle = shuffle
	local loaded_data = torch.load(input_file)
	self.e1 = loaded_data.e1
	self.paths = loaded_data.paths
	self.e2 = loaded_data.e2
	self.e_neg = self:generate_neg_entities()
	self.data_size = self.e1:size(1)
	self.curr_start = 1
	self.useCuda = useCuda
	self.return_neg = return_neg
	self.neg_input_file = neg_input_file
	self.negative_examples = nil
	if self.return_neg then	
		assert(self.neg_input_file ~= nil, 'provide the path to negative entities') 
		assert(self.do_shuffle == false, 'right now shuffle isnt supported with this.')
		-- self.negative_examples_batch = torch.LongTensor(self.batch_size, self.entity_vocab_size)
		print('Reading file of negative examples...')
		self.negative_examples = torch.load(neg_input_file)
		print('Done')
	end
end

function Batcher:generate_neg_entities()
	--randomly sample a negative entity of the same size as e2.
	return torch.rand(self.e2:size()):mul(self.entity_vocab_size):floor():add(1)
end

function Batcher:shuffle()
	local inds = torch.randperm(self.e1:size(1)):long()
	self.e1 = self.e1:index(1, inds)
	self.e2 = self.e2:index(1, inds)
	self.e_neg = self.e_neg:index(1, inds)
	self.paths = self.paths:index(1, inds)
end


function Batcher:get_batch()
	
	local start_index = self.curr_start
	if start_index > self.data_size then return nil end
	local end_index = math.min(start_index+self.batch_size-1, self.data_size)
	local curr_batch_size = end_index - start_index + 1
	local batch_e1 = self.e1:narrow(1, start_index, curr_batch_size)
	local batch_paths = self.paths:narrow(1, start_index, curr_batch_size)
	local batch_e2 = self.e2:narrow(1, start_index, curr_batch_size)
	local batch_e_neg = self.e_neg:narrow(1, start_index, curr_batch_size)
	local neg_examples = nil
	if self.return_neg then
		
	end
	local ret = {}
	if self.return_neg then
		batch_neg_examples, num_neg_examples = self:get_negative_examples_batch(start_index, end_index)
		assert(batch_neg_examples:size(1) == num_neg_examples:size(1))
		ret = {batch_e1, batch_paths, batch_e2, batch_e_neg, batch_neg_examples, num_neg_examples}
	else
		ret = {batch_e1, batch_paths, batch_e2, batch_e_neg}	
	end
	self.curr_start = end_index + 1
	return ret

end
--Different from generate_neg_entities, This method is used at test time
--for a fair comparison with Guu et al '15. These return the precomputed negative examples
-- for each test entity
function Batcher:get_negative_examples_batch(start_index, end_index)
	
	local all_neg_entities = {}
	local orig_sizes = {}
	for i = start_index, end_index do
		--1. get the negative entities tensor - num_negative_entities tensor
		local neg_entities = self.negative_examples[i]
		local num_negative_entities = neg_entities:size(1)
		-- local num_padding = self.entity_vocab_size - num_negative_entities
		-- local neg_entities_padded = nn.Padding(1, num_padding, 1, self.entity_vocab_size+1)(neg_entities) --pad with vocab_size + 1 so that I can make the vocab_size + 1 col to -inf
		-- 2. Create a padded tensor of size self.entity_vocab_size fill it with self.entity_vocab_size+1
		neg_entities_padded = torch.Tensor(self.entity_vocab_size):fill(self.entity_vocab_size+1)
		-- 3. Use scatter to copy the id in neg_entities in the idth position of neg_entities_padded
		neg_entities_padded:scatter(1, neg_entities, neg_entities:double())
		-- 4. Put the index of e2
		neg_entities_padded[self.e2[i]] = self.e2[i]
		neg_entities_padded = neg_entities_padded:view(1, self.entity_vocab_size)
		table.insert(all_neg_entities, neg_entities_padded)
		table.insert(orig_sizes, num_negative_entities)
	end
	return nn.JoinTable(1):forward(all_neg_entities), torch.Tensor(orig_sizes)

end

function Batcher:reset()
	self.curr_start = 1
	-- self.e_neg = self:generate_neg_entities()
	if self.do_shuffle then self:shuffle() end
end

function Batcher:get_data_size() return self.data_size end
