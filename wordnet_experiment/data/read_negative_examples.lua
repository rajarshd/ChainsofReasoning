require 'torch'
require 'os'
require 'Util'
require 'nn'


local neg_input_file = '/home/rajarshi/canvas/traversing_knowledge_graphs/data_formatted/wordnet/test_neg.int'
local neg_output_file = '/home/rajarshi/canvas/traversing_knowledge_graphs/data_formatted/wordnet/test_neg.torch'
local all_neg = {}
counter = 0
for line in io.lines(neg_input_file) do

	local fields = Util:splitByDelim(line,",",true)
	local index = torch.LongTensor(fields)
	index:add(1)
	table.insert(all_neg, index)
	counter = counter + 1
	print(counter)
end
torch.save(neg_output_file, all_neg)

