function [state,metadata] = sppr_segment(X,cfg)
%SPPR_SEGMENT Compute ERS pseudo-labels from the input scene.
assert(strcmp(cfg.segmentation_source,'recompute'),'Unknown segmentation source.');
assert(exist('mex_ers','file')==3, ...
    'ERS MEX missing. Configure mex -setup C++ and run main again.');
[H,W,B]=size(X); K=floor(double(H)*double(W)/cfg.target_area);
assert(K>=2 && K<H*W,'ERS requires 2 <= floor(H*W/target_area) < H*W.');
Y=double(reshape(X,H*W,B)); lo=min(Y,[],1); span=max(Y,[],1)-lo;
span(span==0)=1; Y=(Y-lo)./span;
scores=release_pca(Y,1,true); scores=reshape(scores,H,W);
span=max(scores(:))-min(scores(:));
assert(span>0,'First principal component is constant.');
image=double(uint8(round(255*(scores-min(scores(:)))/span)));
map=mex_ers(image,K,cfg.ers_lambda,cfg.ers_sigma);
found=numel(unique(map));
if found~=K
    warning('SPPR:RegionCount','ERS returned %d regions instead of %d; continuing.',found,K);
end
state=sppr_prepare(X,map);
metadata=struct('K',K,'actual_K',found,'target_area',cfg.target_area, ...
    'lambda',cfg.ers_lambda,'sigma',cfg.ers_sigma,'ground_truth_used',false, ...
    'mex_file',which('mex_ers'));
end

function [scores,basis] = release_pca(Y,dimension,center_projection)
%RELEASE_PCA Base-MATLAB covariance PCA, no external PCA or mapminmax.
if nargin<3, center_projection=true; end
Y=double(Y); mu=mean(Y,1); Z=Y-mu;
[V,D]=eig(Z'*Z,'vector'); [~,order]=sort(real(D),'descend');
basis=real(V(:,order(1:min(dimension,size(Y,2)))));
% Fix otherwise arbitrary signs for reproducible grayscale segmentation.
for k=1:size(basis,2)
    [~,j]=max(abs(basis(:,k))); if basis(j,k)<0, basis(:,k)=-basis(:,k); end
end
if center_projection, scores=Z*basis; else, scores=Y*basis; end
end
