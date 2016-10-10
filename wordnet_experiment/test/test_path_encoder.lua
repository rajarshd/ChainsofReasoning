--Test cases for the RNN network

package.path = package.path ..';../model/?.lua'

require 'PathEncoder'


local input_dim = 250
local relation_vocab_size = 24
local output_dim = 250
local useCuda = true

local p = PathEncoder(input_dim, output_dim, relation_vocab_size)
local path_encoder = p:build_encoder()

local seq_len = 5
local batch_size = 32
local synthetic_data = torch.rand(batch_size, seq_len):mul(relation_vocab_size):floor()

local path_repr = path_encoder(synthetic_data)

--1,. Test whether the size of output is equal to sequence length
print('Test whether the size of output is equal to sequence length')
assert(#path_repr == seq_len)
print('Test passed!')

--2. Test padding
batch_size = 1
synthetic_data = torch.zeros(batch_size, seq_len)
path_repr = path_encoder(synthetic_data)
print('Test whether padding works')
for i=1, #path_repr do
	for j=1, output_dim do
		assert(path_repr[i][1][j] == 0)
	end
end
print('Test passed!')

