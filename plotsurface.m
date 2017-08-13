function [xq,yq,vq]=plotsurface(x,y,v)
gx = min(x):0.01:max(x);
gy =min(y):0.01:max(y);
F = scatteredInterpolant(x,y,v);
[ xq, yq ] = meshgrid(gx, gy);
F.Method = 'linear';
vq = F(xq,yq);
figure
surfc(xq,yq,vq)
title('Test output - Linear Interpolation')
