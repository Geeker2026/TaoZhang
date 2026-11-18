function features = tdlpp_project(blocks,model)
%TDLPP_PROJECT Array form of X x1 U1 x2 U2 x3 U3.
I1=size(blocks{1},1); K=kron(model.U3,model.U2);
features=zeros(numel(blocks),size(model.U1,1)*size(K,1));
for i=1:numel(blocks)
    z=model.U1*reshape(double(blocks{i}),I1,[])*K'; features(i,:)=z(:)';
end
end
