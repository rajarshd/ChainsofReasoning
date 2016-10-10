require 'nn'
require 'rnn'

local PathEncoder = torch.class('PathEncoder')

function PathEncoder:__init(input_dim, output_dim, relation_vocab_size, rnn_type)
	-- body
	self.input_dim = input_dim
	self.output_dim = output_dim
	self.vocab_size = relation_vocab_size
	self.rnn_type = rnn_type
end

function PathEncoder:build_encoder()
	-- body
	local input2hidden = nn.Linear(self.input_dim, self.output_dim)
	local hidden2hidden = nn.Linear(self.output_dim, self.output_dim)
	local nonLinear = nn.ReLU()

	--recurrent module
	local rm = nn.Sequential()
				   :add(nn.ParallelTable()
				      :add(input2hidden)
				      :add(hidden2hidden))
				   :add(nn.CAddTable())
				   :add(nonLinear)
	rm = nn.MaskZero(rm,1) --to take care of padding

	local rnn = nn.Sequencer(nn.Recurrence(rm, self.output_dim, 1))

	local net = nn.Sequential()
					:add(nn.LookupTableMaskZero(self.vocab_size, self.input_dim))
					:add(nn.SplitTable(2))
					:add(rnn)
	return net
end