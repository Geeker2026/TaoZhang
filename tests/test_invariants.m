function report = test_invariants()
%TEST_INVARIANTS Fixed checks on SPPR candidate assignment and TDLPP plumbing.
% Needs no dataset and no ERS build; runs in a few seconds.
root=fileparts(fileparts(mfilename('fullpath')));
saved_path=path; path_cleanup=onCleanup(@()path(saved_path)); %#ok<NASGU>
addpath(fullfile(root,'core'),fullfile(root,'third_party','legacy_pca'),'-begin');
saved_rng=rng; rng_cleanup=onCleanup(@()rng(saved_rng)); %#ok<NASGU>
rng(91,'twister'); checks={}; X=single(rand(7,7,4));

% 1. Unique candidates, exhaustion handling, SoftClip bound.
map=2*ones(7); map(3,3)=1; map(1,1)=1;
[z,info]=sppr_reconstruct(X,sppr_prepare(X,map),3,3,3);
raw=extract_patch(X,3,3,3); assigned=info.source_linear>0;
assert(nnz(assigned)==1 && info.unassigned_count==7);
assert(numel(unique(info.source_linear(assigned)))==nnz(assigned));
assert(isequal(z(repmat(~assigned,1,1,4)),raw(repmat(~assigned,1,1,4))));
for i=1:3
    for j=1:3
        original=double(reshape(raw(i,j,:),1,[]));
        modified=double(reshape(z(i,j,:),1,[]));
        center=double(reshape(raw(2,2,:),1,[]));
        assert(norm(modified-original)<=norm(original-center)+2e-7);
    end
end
checks{end+1}='candidate_uniqueness_exhaustion_and_softclip_bound';

% 2. Empty candidate pool and fully homogeneous patch both return the input.
map(1,1)=2; [z,info]=sppr_reconstruct(X,sppr_prepare(X,map),3,3,3);
assert(isequal(z,raw) && info.unassigned_count==8);
[z,info]=sppr_reconstruct(X,sppr_prepare(X,ones(7)),1,1,5);
assert(isequal(z,extract_patch(X,1,1,5)) && ~any(info.mask(:)));
checks{end+1}='empty_pool_and_homogeneous_identity';

% 3. Symmetric padding at the border.
[~,rr,cc]=extract_patch(X,1,1,5); assert(isequal(rr,[2 1 1 2 3]) && isequal(rr,cc));
checks{end+1}='symmetric_padding';

% 4. Candidate membership partitions the scene and stays in it.
state=sppr_prepare(X,map);
all_members=sort(vertcat(state.members{:}));
assert(isequal(all_members(:),(1:numel(map))'));
assert(strcmp(state.candidate_scope,'original_scene_only'));
checks{end+1}='candidate_membership_partition';

% 5. Within-class graph is symmetric, zero-diagonal, class-blocked, and uses
%    the heat kernel exp(-d^2/t1) with t1 the mean training-pair distance.
blocks=arrayfun(@(~)single(rand(3,3,4)),1:20,'UniformOutput',false);
labels=[10*ones(10,1);30*ones(10,1)];
[W,B,t1,t2]=tdlpp_graph(blocks,labels);
assert(isequal(W,W') && isequal(B,B') && ~any(diag(W)));
assert(~any(W(1:10,11:20),'all') && t1>0 && t2>0);
d=double(norm(blocks{1}-blocks{2},'fro'));
assert(abs(W(1,2)-exp(-d^2/t1))<1e-12 && abs(W(1,2)-W(2,1))<1e-12);
checks{end+1}='within_class_graph_heat_kernel';

% 6. Solver returns finite projections and keeps the frozen history contract.
cfg=struct('dimension',2,'ridge',0,'max_iterations',50,'tolerance',0.005,'t1',[],'t2',[]);
model=tdlpp_fit(blocks,labels,cfg);
assert(isequal(model.classes,[10;30]) && model.iterations<=50);
assert(strcmp(model.history_position,'before_sweep_frozen') && model.ridge==0);
assert(all(isfinite(model.train_features(:))));
assert(size(model.train_features,2)==size(model.U1,1)*size(model.U2,1)*size(model.U3,1));
checks{end+1}='solver_projection_and_history';

report=struct('passed',true,'checks',{checks},'matlab_version',version);
fprintf('PASS: %d invariant checks.\n',numel(checks));
end
