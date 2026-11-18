function [patch,rr,cc] = extract_patch(X,row,col,window)
%EXTRACT_PATCH Symmetric padding without Image Processing Toolbox.
validateattributes(window,{'numeric'},{'scalar','integer','positive','odd'});
[H,W,~]=size(X);
validateattributes(row,{'numeric'},{'scalar','integer','>=',1,'<=',H});
validateattributes(col,{'numeric'},{'scalar','integer','>=',1,'<=',W});
half=(window-1)/2;
rr=reflect(row-half:row+half,H); cc=reflect(col-half:col+half,W);
patch=X(rr,cc,:);
end
function indices=reflect(indices,n)
t=mod(indices-1,2*n)+1; indices=min(t,2*n+1-t);
end
