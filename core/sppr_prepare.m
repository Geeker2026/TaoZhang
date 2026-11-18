function state = sppr_prepare(X,pseudo_map)
%SPPR_PREPARE Candidate membership from ORIGINAL scene only. No ground truth.
[H,W,~]=size(X);
assert(isequal(size(pseudo_map),[H W]) && all(isfinite(pseudo_map(:))), ...
    'Pseudo-map must be finite and match the original scene.');
[ids,~,membership]=unique(pseudo_map(:),'sorted');
members=accumarray(membership,(1:H*W)',[],@(v){v});
state=struct('height',H,'width',W,'pseudo_map',pseudo_map, ...
    'ids',ids,'members',{members},'candidate_scope','original_scene_only', ...
    'algorithm','unique_candidate_greedy_softclip_frozen_v2');
end
