local LookupTableWithGrad, parent = torch.class('nn.LookupTableWithGrad', 'nn.LookupTable')

function LookupTableWithGrad:__init(nIndex, nOutput)
   parent.__init(self,nIndex,nOutput)
end

function LookupTableWithGrad:updateGradInput(input,gradInput)
	self.gradInput:resizeAs(input)
	self.gradInput:zero()
	return self.gradInput
end