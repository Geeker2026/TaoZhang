function [reconstructed,info] = sppr_reconstruct(X,state,row,col,window)
%SPPR_RECONSTRUCT Frozen row-major unique-candidate assignment + SoftClip.
% Used candidates are excluded; exhaustion retains remaining raw pixels.
% This is sequential greedy assignment, not independent position-wise argmin.
[raw,rr,cc]=extract_patch(X,row,col,window);
map=state.pseudo_map; mask=map(rr,cc)~=map(row,col);
half=(window-1)/2; ref_rows=row-half:row+half; ref_cols=col-half:col+half;
[rg,cg]=ndgrid(ref_rows,ref_cols);
% Match frozen padded coordinates. Out-of-scene pure positions do not
% exclude their reflected original-scene counterpart from the pool.
inside=rg>=1 & rg<=state.height & cg>=1 & cg<=state.width;
pure=~mask & inside;
pure_linear=sub2ind([state.height state.width],rg(pure),cg(pure));
slot=find(state.ids==map(row,col),1);
candidates=state.members{slot}; candidates=candidates(~ismember(candidates,pure_linear));
hard=raw; source=zeros(window,window); rho=mean(mask(:)); h=half+1;
if ~isempty(candidates) && any(mask(:))
    cube=reshape(X,state.height*state.width,[]);
    center=double(reshape(raw(h,h,:),1,[]));
    spectral=sqrt(sum((double(cube(candidates,:))-center).^2,2));
    [cr,cl]=ind2sub([state.height state.width],candidates);
    used=false(numel(candidates),1); [hj,hi]=find(mask');
    for t=1:numel(hi)
        available=find(~used); if isempty(available), break; end
        spatial=sqrt((double(cr(available))-ref_rows(hi(t))).^2 + ...
            (double(cl(available))-ref_cols(hj(t))).^2);
        scores=spatial./exp(-spectral(available));
        [~,local]=min(scores); best=available(local);
        hard(hi(t),hj(t),:)=X(cr(best),cl(best),:);
        source(hi(t),hj(t))=candidates(best); used(best)=true;
    end
end
delta=hard-raw;
distance=sqrt(sum((double(raw)-double(raw(h,h,:))).^2,3));
magnitude=sqrt(sum(double(delta).^2,3));
alpha=single(min(1-rho,distance./(magnitude+eps)));
reconstructed=raw+alpha.*delta;
info=struct('mask',mask,'rho',rho,'alpha',alpha,'source_linear',source, ...
    'hard_assignment',hard,'candidate_count',numel(candidates), ...
    'unassigned_count',nnz(mask & source==0),'algorithm',state.algorithm);
end
