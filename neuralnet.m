function    [errors, performance] = neuralnets(X,T,hiddenunits);
net = feedforwardnet(hiddenunits,'trainlm');
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
%net.numLayers=1:10;
[ net, tr] = train(net,X,T); 
outputs = net(X);
errors = gsubtract(T, outputs);
performance = perform(net, T, outputs);
view(net)
