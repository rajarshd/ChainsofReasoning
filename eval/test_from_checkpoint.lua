require 'torch'
require 'nn'
require 'optim'
require 'rnn'
require 'os'
require 'cunn'
package.path = package.path ..';./model/batcher/?.lua'
package.path = package.path ..';../model/batcher/?.lua'
package.path = package.path ..';./model/module/?.lua'
package.path = package.path ..';../model/module/?.lua'
require 'MapReduce'
require 'BatcherFileList'
require 'SplitTableNoGrad'
require "ConcatTableNoGrad"
require "TopK"
require "LogSumExp"


cmd = torch.CmdLine()


cmd:option('-output_scores',0,'output scores for MAP scores')
cmd:option('-input_dir','','input dir which contains the list files for train/dev/test')
cmd:option('-out_dir','','output dir for outputing score files')
cmd:option('-predicate_name','','output dir for outputing score files')
cmd:option('-meanModel',0,'to take the mean of scores of each path. Default is max (0)')
cmd:option('-gpuid',3,'which gpu to use. -1 = use CPU')
cmd:option('-epochs_file','','epochs to test')
cmd:text()
local params = cmd:parse(arg)

local models = {}

local output_scores = params.output_scores==1
local out_dir = params.out_dir
local input_dir = params.input_dir
local predicate_name = params.predicate_name
local epochs_file = params.epochs_file

--read the epochs to evaluate on
epoch_counter = 0
for line in io.lines(epochs_file) do
	models[epoch_counter + 1] = line
	epoch_counter = epoch_counter + 1
end

require 'cutorch'
require('cunn')
cutorch.setDevice(params.gpuid + 1) 

assert(input_dir~='','input_dir isnt set. Point to the dir where train/dev/test.list files reside')
if(output_scores) then
	assert(out_dir~='' and predicate_name~='','if output_scores is set, then please specify an out_dir and predicate_name')
end

data_files={input_dir..'/test.list',input_dir..'/dev.list'}
local shuffle = false
local maxBatches = 150
local minibatch = 32
local useCuda = true
local testBatcher = BatcherFileList(data_files[1], minibatch, shuffle, maxBatches, useCuda)
local devBatcher = BatcherFileList(data_files[2], minibatch, shuffle, maxBatches, useCuda)
local testBatchCounter = 0 --count total number of test batches
local devBatchCounter = 0 --count total number of dev batches

while (true) do
	local labs,inputs,count,classId = testBatcher:getBatch()
	if(inputs == nil) then break end
	testBatchCounter = testBatchCounter + 1
end
while (true) do
	local labs,inputs,count,classId = devBatcher:getBatch()
	if(inputs == nil) then break end
	devBatchCounter = devBatchCounter + 1
end

for i =1, epoch_counter do
	model_name = models[i]
	print(model_name)
	local model_path = out_dir..'/'..model_name
	local out_dir_p = out_dir..'/'..predicate_name
	assert(model_path~='','model path isnt set')
	print('Loading saved model from..'..model_path)
	predicate_name_to_print = predicate_name..'_'..model_name
	local check_file = io.open(model_path)
	if check_file ~= nil then
		local predictor_net = torch.load(model_path)
		-- local predictor_net = checkpoint.predictor_net
		local reducer = nil
		if(params.meanModel == 1) then
			reducer = nn.Mean(2)
			print('Mean model!')
		elseif (params.meanModel == 0) then
			reducer = nn.Max(2)
			print('Max model!')
		elseif (params.meanModel == 2) then
			reducer = nn.Sequential():add(nn.TopK(5,2)):add(nn.Mean(2))
			print('TopK model!')
		elseif (params.meanModel == 3) then
			print('Reducer is LogSumExp')
			reducer = nn.Sequential():add(nn.LogSumExp(2)):add(nn.Squeeze(2))
		end
	
		--prediction_net = nn.Sequential():add(predictor_net):add(nn.Sigmoid())
		prediction_net = nn.MapReduce(predictor_net,reducer)
		prediction_net = nn.Sequential():add(prediction_net):add(nn.Sigmoid())
		prediction_net = prediction_net:cuda()
		prediction_net:evaluate() -- turn on evaluate flag; imp when using dropout
		--data_files={input_dir..'/dev.list'}
		out_files={out_dir_p..'/test.scores.'..model_name,out_dir_p..'/dev.scores.'..model_name} --maintain the sequence of test/train/dev between data and out files table.
		--out_files={out_dir..'/dev.scores'}
		acc_file = nil
		if(output_scores) then
			accuracy_file=out_dir_p..'/accuracy.txt'
			acc_file = io.open(accuracy_file,'w')
		end
		lazyCuda = false
		numRowsToGPU = 20
		print('STARTING EVALUATION...')
		if(output_scores) then
			acc_file:write('STARTING EVALUATION...\n')
		end
		for d=1,#data_files do
			print('Procesing '..data_files[d])
			if(output_scores) then
				acc_file:write('Procesing '..data_files[d]..'\n')
			end
			list = data_files[d]
			local totalBatchCounter
			if d == 1 then
				batcher = testBatcher
				totalBatchCounter = testBatchCounter
			else
				batcher = devBatcher
				totalBatchCounter = devBatchCounter
			end
			batcher:reset()
			out_file=''
			file=nil
			if(output_scores) then
				out_file = out_files[d]
				file = io.open(out_file, "w")
			end
			total_correct = 0
			total_count = 0
			counter=0
			local batch_counter = 0
			while(true) do
				local labs,inputs,count,classId = batcher:getBatch()		
				if(inputs == nil) then break end
				labs = labs:cuda()
				inputs = inputs:cuda()
				batch_counter = batch_counter + 1
				local preds = nn.Sequential():add(prediction_net):add(nn.Select(2,classId)):cuda():forward(inputs)
				if(output_scores) then
					for i=1,count do
						score = preds[i]
						label = labs[i]
						file:write(predicate_name_to_print..'\t'..counter..'\t'..score..'\t'..label..'\n')
						counter = counter + 1
					end
				end
				-- preds,pi = torch.max(preds,2)
				pi = preds:gt(0.5):typeAs(labs)
				local correct = pi:eq(labs):sum()
				total_correct = total_correct + correct
				total_count = count + total_count
				xlua.progress(batch_counter, totalBatchCounter)
			end	
			if(output_scores) then
				file:close()
			end
			correct = 100*total_correct/total_count
			print(correct..'%')
			if(output_scores) then
				acc_file:write(correct..'%\n')
			end
			print('Processed '..total_count..' examples')
			if(output_scores) then
				acc_file:write('Processed '..total_count..' examples\n')
			end
		end
		
		-- prediction_net = prediction_net:double()

		if(output_scores) then
			acc_file:close()
		end
	end
end