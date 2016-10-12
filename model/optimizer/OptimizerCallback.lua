local OptimizerCallback = torch.class('OptimizerCallback')

--here, 'hook' is a function (i), where is is the epoch that the optimizer is currently at. 
--This is useful, for example, for saving models with different names depending on the epoch

function OptimizerCallback:__init(epochHookFreq,hook,name)
	self.epochHookFreq = epochHookFreq
	self.hook = hook
	self.name = name
end

