function    [trainval_errors, trainval_performance, test_Perform,test_Errors, tr, test_Outputs net] = neuralnets(X,T,hiddenunits,Test_input,Test_Target)
net = feedforwardnet(hiddenunits,'trainlm');
net.layers{1}.transferFcn = 'tansig';
%net.numLayers=1:10;
[ net, tr] = train(net,X,T); 
trainval_outputs = net(X);
trainval_errors = gsubtract(T, trainval_outputs);
trainval_performance = perform(net, T, trainval_outputs);
test_Outputs = net(Test_input);
test_Perform = perform(net, Test_Target, test_Outputs);
test_Errors = gsubtract(Test_Target, test_Outputs);
 