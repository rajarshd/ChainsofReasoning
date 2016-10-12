--takes in a data file and implements logic for returning batches from it.
local Batcher = torch.class('Batcher')

function  Batcher:__init(fileName, batchSize, shuffle)
	
	local loadedData = torch.load(fileName)
	self.labels = loadedData.labels
	self.data = loadedData.data
	self.classId = loadedData.classId
	self.doShuffle = shuffle
	if (self.labels:dim() == 1) then
		self.labelDimension = 1	
		self.labels = self.labels:mul(-1):add(2)
	else
		self.labelDimension = self.labels:size(2)
	end
	self.numPaths = self.data:size(2)
	self.numTokensInPath = self.data:size(3)
	self.numFeatureTemplates = self.data:size(4)
	if self.doShuffle then self:shuffle() end --first shuffle
	self.batchSize = batchSize
	self.curStart = 1
end


function Batcher:shuffle()
	if(self.doShuffle) then
		local inds = torch.randperm(self.labels:size(1)):long()
		self.labels = self.labels:index(1,inds)
		self.data = self.data:index(1,inds)
	end
end

function Batcher:getBatch()

	local dataSize = self.labels:size(1)
	local startIndex = self.curStart
	if startIndex > dataSize then return nil end
	local endIndex = math.min(startIndex+self.batchSize-1, dataSize)
	local currBatchSize = endIndex - startIndex + 1
	local batchLabels = self.labels:narrow(1, startIndex, currBatchSize)
	local batchData = self.data:narrow(1, startIndex, currBatchSize)
	self.curStart = endIndex + 1
	return batchLabels, batchData
end

function Batcher:reset()
	self.curStart = 1
	if self.doShuffle then self:shuffle() end
end

--return all the dimensions except the first dimension of the data because they are fixed and will be used to preallocate tensors in gpu.
function Batcher:getSizes() return self.labelDimension, self.numPaths, self.numTokensInPath, self.numFeatureTemplates end

--return the classId associated with this batcher
function Batcher:getClassId() return self.classId end