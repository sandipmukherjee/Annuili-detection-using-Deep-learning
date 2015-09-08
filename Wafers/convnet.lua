require 'fs'
require 'csv'
require 'nn'
require 'image'
require 'optim'

dataDir = "/Users/mrueg/Desktop/TI_preprocessed"
if arg then
   dataDir = arg[1] -- should be the directory that contains the *preprocessed* dataset
end
pSep = "/"
batchDirs = {}
inputSize = 1732
isBad = true -- labels are boolean where true indicates a bad wafer

-- find the absolute path of the directory for all batches
for _,chValue in ipairs(fs.readdir(dataDir)) do
        local chipDir = dataDir .. pSep .. chValue
        if chValue ~= ".." and fs.is_dir(chipDir) then
               -- if string.sub(chipDir,0,1) == "_" then -- case for _mixed "batch"
               for _,bValue in ipairs(fs.readdir(chipDir)) do
                      local batchDir = chipDir .. pSep .. bValue
                      if bValue ~= ".." and fs.is_dir(batchDir) then
                             table.insert(batchDirs, batchDir)
                      end
               end
        end
end

-- split data into training/validation/test set using a 60/20/20 split and a fixed seed for repeatable experiments
torch.manualSeed(42)
shuffle = torch.randperm(#batchDirs)

trainDirs = {}
validationDirs = {}
testDirs = {}
trainSplit = math.ceil(#batchDirs * 0.6)
validationSplit = math.ceil(#batchDirs * 0.8)
testSplit = #batchDirs
for i=1,trainSplit do
        table.insert(trainDirs, batchDirs[ shuffle[i] ])
end
for i=(trainSplit + 1), validationSplit do
        table.insert(validationDirs, batchDirs[ shuffle[i] ])
end
for i=validationSplit + 1, testSplit do
        table.insert(testDirs, batchDirs[ shuffle[i] ])
end
        
-- data loading functions
normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7)):float() -- TODO: maybe use slightly larger number (e.g 15)?

function batchMedian(batchPath)
        local img = image.load(batchPath .. pSep .. "median.jpg")
        local hsv = image.rgb2hsv(img)
        hsv = hsv:float()
        hsv[3] = normalization(hsv[{{3}}]) -- normalization, but only for brightness
        local scaledImg = image.scale(hsv, inputSize, inputSize, 'bilinear')
        return scaledImg
end

function trim(s)
  return s:match'^%s*(.*%S)' or ''
end
        
-- load a batch, based on a batch path. usable in a for .. in loop
function loadBatch(batchPath)
        local indx = 1
        local files = fs.readdir(batchPath)
        local labels = {}
        for _,label in ipairs(csv.load(batchPath .. pSep .. "labels.csv", ",", "raw", false, false)) do
			   if trim(label[1]) == "1" then
				   labels[trim(label[2])] = 1
			   else
				   labels[trim(label[2])] = 0
			   end 
        end
        
        
        return function()
               local imgName = files[indx]
               if not imgName then return nil end
               while imgName == ".." or imgName == "median.jpg" or string.sub(imgName,-4) ~= ".jpg" do
                      indx = indx + 1
                      imgName = files[indx]
                      if not imgName then return nil end
               end
               if not imgName then return nil end
               local imgNameRaw = string.sub(imgName,0,-5) -- image name without extension
               local img = image.load(batchPath .. pSep .. imgName)
               local hsv = image.rgb2hsv(img)
               local hsv = hsv:float()
               -- whitening, but only on brightness
               -- see http://torch.cogbits.com/doc/tutorials_supervised/index.html for why exactly we do this.
               -- local yuv = image.rgb2yuv(img)
               -- yuv[1] = normalization(yuv[{{1}}])
               hsv[3] = normalization(hsv[{{3}}])
               local scaledImg = image.scale(hsv, inputSize, inputSize, 'bilinear')
               -- print("ready" .. imgNameRaw .. " | " .. labels[imgNameRaw])
               
               indx = indx + 1
               
               return { ["img"] = scaledImg, ["label"] = labels[imgNameRaw], ["name"] = imgNameRaw }
               -- TODO: also supply the mask.
        end
end


-- build the model
network = nn.Sequential()
-- the convolutional part needs 3D arrays.
-- pass in a 2xsizexsize array to allow for passing in both the current image and the batch median
network:add(nn.Reshape(2,3,inputSize,inputSize)) 

do
    meanAndImage = nn.Parallel(1,1)
    -- 2 convolution filters in parallel, once for mean, once for current image
    
    inputFilter = nn.SpatialConvolution(3,3,5,5);
    inputFilter2 = inputFilter:clone('weight','bias'); -- weight sharing
    meanAndImage:add(inputFilter)
    meanAndImage:add(inputFilter2)
    
    network:add(meanAndImage)
end

network:add(nn.SpatialConvolution(2*3,3,3,3)) -- output dim now: 3*1726*1726
-- 
-- -- inner layers. the sizes in these are basically a matter of taste
network:add(nn.SpatialConvolution(3,3,7,7))
network:add(nn.Tanh())
network:add(nn.SpatialMaxPooling(4,4,4,4))

network:add(nn.SpatialConvolution(3,16,7,7))
network:add(nn.Tanh())
network:add(nn.SpatialMaxPooling(4,4,4,4))

network:add(nn.SpatialConvolution(16,4,7,7))
network:add(nn.Tanh())
network:add(nn.SpatialMaxPooling(4,4,4,4))

-- -- TODO: add some max pooling and convolution layers
network:add(nn.Reshape(4*25*25))
network:add(nn.Linear(4*25*25,2))
network:add(nn.LogSoftMax()) -- could use Sigmoid instead

-- do
-- 	parameters,gradParameters = network:getParameters()
-- 	local i = 1
--     local median = batchMedian(trainDirs[i])
--     local inputs={}
--     local targets={}
--     
--    print("Handling batch " .. trainDirs[i])
--     for val in loadBatch(trainDirs[i]) do
--         local img = val.img
--         local label = val.label
--         local name = val.name
-- 		local input = torch.Tensor(2,3,inputSize,inputSize)
-- 		input[1] = img
-- 		input[2] = median
--         table.insert(inputs,input)
--         table.insert(targets,label)
--     end
-- 	print "build"
-- 	for i = 1,#inputs do
-- 		local output = network:forward(inputs[i])
-- 		print(output) -- :size())
-- 		break
-- 	end
-- 	print "done"
-- end


-- training
criterion = nn.ClassNLLCriterion()
confusion = optim.ConfusionMatrix(2)

parameters,gradParameters = network:getParameters()

function train(dataset)
   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()

   -- shuffle training data at each epoch
   shuffle = torch.randperm(#trainDirs)

   print("Training at epoch " .. epoch)

   for i = 1,#trainDirs do
       local median = batchMedian(trainDirs[shuffle[i]])
       local inputs={}
       local targets={}
       
	   print("Handling batch " .. trainDirs[shuffle[i]])
       for val in loadBatch(trainDirs[shuffle[i]]) do
           local img = val.img
           local name = val.name
	   	   local input = torch.Tensor(2,3,inputSize,inputSize)
	   	   input[1] = img
	   	   input[2] = median
           table.insert(inputs,input)
		   local label = val.label + 1
           table.insert(targets,label)
       end
       
      -- disp progress
      xlua.progress(i, #trainDirs)

      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
          -- get new parameters
          if x ~= parameters then
              parameters:copy(x)
          end

          -- reset gradients
          gradParameters:zero()

          -- f is the average of all criterions
          local f = 0

          -- evaluate function for complete mini batch
          for i = 1,#inputs do
              -- estimate f
              local output = network:forward(inputs[i])
              local err = criterion:forward(output, targets[i])
              f = f + err
              
              -- estimate df/dW
              local df_do = criterion:backward(output, targets[i])
              network:backward(inputs[i], df_do)

              -- update confusion
              confusion:add(output, targets[i])
          end
          
          -- normalize gradients and f(X)
          gradParameters:div(#inputs)
          f = f/#inputs
          
          -- return f and df/dX
          return f,gradParameters
      end
      -- 
      -- config = config or {learningRate = 0.1,
      --                     weightDecay = 0.01,
      --                     momentum = 0.6,
      --                     learningRateDecay = 5e-7}
      -- optim.sgd(feval, parameters, config)
      -- 	  
      config = config or {learningRate = 0.1,
                          weightDecay = 0.01,
                          momentum = 0.6,
                          learningRateDecay = 5e-7}
	  optim.rprop(feval, parameters, config)
  end
	  
  -- time taken
  time = sys.clock() - time
  time = time / dataset:size()
  print("Time elapsed: " .. (time*1000) .. 'ms')

  -- print confusion matrix
  print(confusion)
  confusion:zero()

  -- next epoch
  epoch = epoch + 1
end

function test(dataset)
    -- local vars
    local time = sys.clock()
  
    -- test over given dataset
    print('Running on testset')
    for t = 1,testDirs:size() do
       -- disp progress
       xlua.progress(t, dataset:size())
	   
       local median = batchMedian(testDirs[t])
	   
       local inputs={}
       local targets={}
       
       for val in loadBatch(trainDirs[shuffle[i]]) do
           local img = val.img
           local label = val.label
           local name = val.name
		   
		   local input = {img,median}
		   local target = label
  
	       -- test sample
	       local pred = network:forward(input)
	       confusion:add(pred, target)
	    end
	end
  
    -- timing
    time = sys.clock() - time
    time = time / testDirs:size()
    print("Time elapsed: " .. (time*1000) .. 'ms')
  
    -- print confusion matrix
    print(confusion)
    confusion:zero()
end

while true do
   train(trainData)
   test(testData)
end