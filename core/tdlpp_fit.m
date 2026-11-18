function model = tdlpp_fit(blocks,labels,cfg)
%TDLPP_FIT Frozen eigensolver, pair order and pre-update stopping behavior.
assert(cfg.ridge==0 && cfg.max_iterations==50 && cfg.tolerance==0.005, ...
    'Frozen core requires ridge=0, max_iterations=50, tolerance=0.005.');
[W,B,t1,t2]=tdlpp_graph(blocks,labels,cfg.t1,cfg.t2);
[classes,~,group]=unique(labels(:),'sorted');
[U1,U2,U3,J,Ci,Cb]=frozen_tdlpp_core(blocks,group,cfg.dimension,W,B);
written=(J~=0)|(Ci~=0)|(Cb~=0); t=find(written,1,'last');
assert(~isempty(t) && all(written(1:t)) && t<=50,'Invalid frozen history.');
reason='max_iterations'; updates=t;
if t>1 && abs(J(t)-J(t-1))<cfg.tolerance
    reason='objective_tolerance'; updates=t-1;
end
model=struct('U1',U1,'U2',U2,'U3',U3,'t1',t1,'t2',t2, ...
    'classes',classes,'ridge',0,'objective',J(1:t),'within',Ci(1:t), ...
    'between',Cb(1:t),'iterations',t,'completed_sweeps',updates, ...
    'effective_dimension',size(U3,1),'requested_dimension',cfg.dimension, ...
    'stop_reason',reason,'history_position','before_sweep_frozen', ...
    'solver','frozen_tdlpp_core');
model.train_features=tdlpp_project(blocks,model);
end

function [U1,U2,U3,J_values,Ci_values,Cb_values] = frozen_tdlpp_core(trainBlocks,trainLabels,dim,Wc,Wb)
%TDLPP_FAST_EQUIV Array implementation of the frozen Code2 TDLPP core.
% Preserves its objective, upper-triangular Wc/Wb convention, class means,
% initialization, eig(BB,A) solve, eigenvalue selection and early stopping.
% It only avoids Tensor Toolbox objects and zero-weight pair evaluations.

threshold=0.005; max_iter=50; t_max=100;
N=numel(trainBlocks); [I1,I2,I3]=size(trainBlocks{1});
labels=trainLabels(:)'; C=numel(unique(labels));
X=zeros(I1,I2,I3,N);
for i=1:N, X(:,:,:,i)=double(trainBlocks{i}); end
F=zeros(I1,I2,I3,C);
for c=1:C
    idx=find(labels==c);
    F(:,:,:,c)=sum(X(:,:,:,idx),4)/numel(idx);
end
[ei,ej,ew]=upper_edges_oldorder(Wc);
[ci,cj,cw]=upper_edges_oldorder(Wb);
U1=eye(1,I1); U2=eye(1,I2); U3=eye(dim,I3);
J_values=zeros(t_max,1); Ci_values=zeros(t_max,1); Cb_values=zeros(t_max,1);
for t=1:max_iter
    Y=project_all(X,U1,U2,U3);
    Yc=project_all(F,U1,U2,U3);
    Ci_values(t)=edge_objective(Y,ei,ej,ew);
    Cb_values(t)=edge_objective(Yc,ci,cj,cw);
    J_values(t)=Ci_values(t)/Cb_values(t);
    if t>1 && abs(J_values(t)-J_values(t-1))<threshold
        fprintf('Converged at iteration t = %d\n',t);
        break;
    end
    [A,BB]=mode_matrices(X,F,U2,U3,ei,ej,ew,ci,cj,cw,1);
    U1=solve_old(BB,A,1);
    [A,BB]=mode_matrices(X,F,U1,U3,ei,ej,ew,ci,cj,cw,2);
    U2=solve_old(BB,A,1);
    [A,BB]=mode_matrices(X,F,U1,U2,ei,ej,ew,ci,cj,cw,3);
    U3=solve_old(BB,A,dim);
end
end

function [ei,ej,ew]=upper_edges_oldorder(W)
N=size(W,1); ei=[]; ej=[]; ew=[];
for i=1:N
    for j=i+1:N
        if W(i,j)~=0, ei(end+1,1)=i; ej(end+1,1)=j; ew(end+1,1)=W(i,j); end %#ok<AGROW>
    end
end
end

function Y=project_all(X,U1,U2,U3)
[I1,~,~,N]=size(X); K=kron(U3,U2); Y=zeros(size(U1,1)*size(U2,1)*size(U3,1),N);
for i=1:N, z=U1*reshape(X(:,:,:,i),I1,[])*K'; Y(:,i)=z(:); end
end

function value=edge_objective(Y,ei,ej,ew)
value=0;
for e=1:numel(ew), d=Y(:,ei(e))-Y(:,ej(e)); value=value+ew(e)*(d'*d); end
end

function [A,BB]=mode_matrices(X,F,Ua,Ub,ei,ej,ew,ci,cj,cw,mode)
[I1,I2,I3,N]=size(X); C=size(F,4);
switch mode
    case 1
        In=I1; K=kron(Ub,Ua); M=zeros(I1,size(K,1),N); Mc=zeros(I1,size(K,1),C);
        for i=1:N, M(:,:,i)=reshape(X(:,:,:,i),I1,[])*K'; end
        for i=1:C, Mc(:,:,i)=reshape(F(:,:,:,i),I1,[])*K'; end
    case 2
        In=I2; K=kron(Ub,Ua); M=zeros(I2,size(K,1),N); Mc=zeros(I2,size(K,1),C);
        for i=1:N, z=permute(X(:,:,:,i),[2 1 3]); M(:,:,i)=reshape(z,I2,[])*K'; end
        for i=1:C, z=permute(F(:,:,:,i),[2 1 3]); Mc(:,:,i)=reshape(z,I2,[])*K'; end
    case 3
        In=I3; K=kron(Ub,Ua); M=zeros(I3,size(K,1),N); Mc=zeros(I3,size(K,1),C);
        for i=1:N, z=permute(X(:,:,:,i),[3 1 2]); M(:,:,i)=reshape(z,I3,[])*K'; end
        for i=1:C, z=permute(F(:,:,:,i),[3 1 2]); Mc(:,:,i)=reshape(z,I3,[])*K'; end
end
A=zeros(In); BB=zeros(In);
for e=1:numel(ew), d=M(:,:,ei(e))-M(:,:,ej(e)); A=A+ew(e)*(d*d'); end
for e=1:numel(cw), d=Mc(:,:,ci(e))-Mc(:,:,cj(e)); BB=BB+cw(e)*(d*d'); end
end

function U=solve_old(BB,A,dk)
[V,D]=eig(BB,A); [lambda,order]=sort(diag(D),'descend'); keep=find(lambda>eps);
if numel(keep)<dk, selected=keep(:); else, selected=keep(1:dk); end
U=V(:,order(selected))';
end
