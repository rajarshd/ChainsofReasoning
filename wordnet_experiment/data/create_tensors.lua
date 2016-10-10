require 'torch'
require 'os'
require 'Util'
require 'nn'
--Script to convert the integer files to lua tables and store.


cmd = torch.CmdLine()
cmd:option('-input_dir','','input dir')
cmd:option('-output_dir','','output dir')


local params = cmd:parse(arg)

local input_dir = params.input_dir
local output_dir = params.output_dir

dirs = {'wordnet', 'freebase'}


input_files = {'train.int', 'dev.int', 'test.int', 'test_kbc.int', 'dev_kbc.int'}
output_files = {'train.torch', 'dev.torch', 'test.torch', 'test_kbc.torch', 'dev_kbc.torch'} --will contain lua tables
for dir_counter, dir_name in pairs(dirs) do
	print(dir_name)
	for file_counter, file_name in pairs(input_files) do
		local e1_table, e2_table, paths_table = {}, {}, {}
		local e1_tensor_table, e2_tensor_table, paths_tensor_table = {}, {},  {}
		print(file_name)
		input_file = input_dir..'/'..dir_name..'/'..file_name
		output_file = input_dir..'/'..dir_name..'/'..output_files[file_counter]
		for line in io.lines(input_file) do
			local fields = Util:splitByDelim(line,"\t",false)
			local e1 = tonumber(fields[1])
			local e2 = tonumber(fields[3])
			local relations = Util:splitByDelim(fields[2],",",true)
			table.insert(e1_table, e1)
			table.insert(e2_table, e2)
			table.insert(paths_table, relations)

			-- ok Luajit doesn't allow tables to go over 2G, so I will have to handle it http://stackoverflow.com/questions/35155444/why-is-luajits-memory-limited-to-1-2-gb-on-64-bit-platforms
			if #e1_table == 100000 then
				e1_tensor = torch.Tensor(e1_table)
				e2_tensor = torch.Tensor(e2_table)
				paths_tensor = torch.Tensor(paths_table)
				table.insert(e1_tensor_table, e1_tensor)
				table.insert(e2_tensor_table, e2_tensor)
				table.insert(paths_tensor_table, paths_tensor)
				e1_table, e2_table, paths_table = {}, {}, {} -- reinitialize the table
			end
		end
		if #e1_table ~= 0 then
			e1_tensor = torch.Tensor(e1_table)
			e2_tensor = torch.Tensor(e2_table)
			paths_tensor = torch.Tensor(paths_table)
			table.insert(e1_tensor_table, e1_tensor)
			table.insert(e2_tensor_table, e2_tensor)
			table.insert(paths_tensor_table, paths_tensor)
			e1_table, e2_table, paths_table = {}, {}, {} -- reinitialize the table
		end

		e1_tensor = nn.JoinTable(1)(e1_tensor_table)
		e2_tensor = nn.JoinTable(1)(e2_tensor_table)
		paths_tensor = nn.JoinTable(1)(paths_tensor_table)

		--since Lua is 1 indexed, just add 1 to everything
		e1_tensor:add(1)
		e2_tensor:add(1)
		paths_tensor:add(1)
		
		assert(e1_tensor:size(1) == e2_tensor:size(1) and e2_tensor:size(1) == paths_tensor:size(1))

		local out = {
		e1 = e1_tensor,
		paths = paths_tensor,
		e2 = e2_tensor
		}
		print('Saving output file '..output_file)
		torch.save(output_file, out)
	end

end