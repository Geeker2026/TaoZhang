function X = preprocess_cube(X,cfg)
%PREPROCESS_CUBE Frozen global normalization/PCA, single-precision output.
X=double(X); assert(all(isfinite(X(:))),'Nonfinite input cube.');
X=normalize(X); [H,W,B]=size(X);
if cfg.pca_dimension>0
    % Preserve the original PCA basis algorithm, signs and uncentered projection.
    basis=frozen_pca_basis(reshape(X,H*W,B),struct('ReducedDim',cfg.pca_dimension));
    X=reshape(reshape(X,H*W,B)*basis,H,W,[]);
end
X=single(normalize(X));
end
function X=normalize(X)
lo=min(X(:)); span=max(X(:))-lo;
assert(span>0,'Constant input cube.'); X=(X-lo)/span;
end
