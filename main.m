function [results,run_dir] = main(dataset,data_root,output_root,seeds,split_file)
%MAIN Single entry for paired TDLPP / SPPR-v2 + TDLPP classification.
% main                           synthetic demonstration (not a benchmark)
% main('Indian','path/to/data')   one real-data trial, recomputed ERS
% main('Salinas','path/to/data','results',1001:1010)  ten paired trials
% main('Salinas','path/to/data','results',[1001 1003],'data/Salinas_splits.mat')
%                                reuse saved index splits instead of resampling
root=fileparts(mfilename('fullpath'));
saved_path=path; path_cleanup=onCleanup(@()path(saved_path)); %#ok<NASGU>
addpath(root,fullfile(root,'core'),fullfile(root,'third_party','ers'), ...
    fullfile(root,'third_party','legacy_pca'),'-begin');
if nargin<1 || isempty(dataset), dataset='demo'; end
if nargin<2 || isempty(data_root), data_root=fullfile(root,'data'); end
if nargin<3 || isempty(output_root), output_root=fullfile(root,'results'); end
if nargin<4 || isempty(seeds)
    if strcmpi(dataset,'demo'), seeds=17; else, seeds=1001; end
end
if nargin<5, split_file=''; end
validateattributes(seeds,{'numeric'},{'vector','integer','nonnegative','finite'});
assert(numel(unique(seeds))==numel(seeds),'Seeds must be unique.');
assert(exist('fitcknn','file')==2, ...
    'Statistics and Machine Learning Toolbox is required (fitcknn).');
cfg=configuration(dataset); cfg.seeds=seeds;
if strcmpi(dataset,'demo')
    saved_rng=rng; rng_cleanup=onCleanup(@()rng(saved_rng)); %#ok<NASGU>
    rng(17,'twister'); H=18; W=18; B=5;
    gt=ones(H,W); gt(:,10:end)=2; X=rand(H,W,B)*0.15;
    for b=1:B, X(:,:,b)=X(:,:,b)+0.35*(gt==2)*(b/B); end
    X=preprocess_cube(X,cfg);
    % A deliberately simple synthetic region map; this demo does not run ERS.
    map=ones(H,W); map(:,10:end)=2; state=sppr_prepare(X,map);
    segmentation=struct('source','synthetic_region_map_not_ERS', ...
        'note','Synthetic example only; not a classification benchmark.');
    source=struct('type','synthetic');
else
    cube_file=fullfile(data_root,cfg.folder,cfg.cube_file);
    gt_file=fullfile(data_root,cfg.folder,cfg.gt_file);
    assert(isfile(cube_file) && isfile(gt_file), ...
        'Missing data files. See README.md for filenames and variables.');
    a=load(cube_file,cfg.cube_variable); b=load(gt_file,cfg.gt_variable);
    X=preprocess_cube(a.(cfg.cube_variable),cfg);
    gt=double(b.(cfg.gt_variable)); clear a b
    ensure_ers(root);
    [state,segmentation]=sppr_segment(X,cfg);
    source=struct('cube_file',cube_file,'gt_file',gt_file,'type','external_data');
end
assert(isequal(size(gt),size(X,[1 2])),'Cube/label dimensions differ.');
assert(all(isfinite(gt(:))) && all(gt(:)>=0) && all(gt(:)==floor(gt(:))), ...
    'Labels must be nonnegative integers; zero means unlabeled.');
assert(numel(unique(gt(gt>0)))>=2,'At least two labeled classes are required.');
% Split definitions: an explicit split file (fifth argument) wins; otherwise the
% packaged data/splits/<Dataset>_splits.mat is reused when it exists.  Seeds
% absent from the file fall back to make_split (same generator, same partitions).
if isempty(split_file)
    split_file=fullfile(data_root,'splits',[cfg.dataset '_splits.mat']);
    if ~isfile(split_file)
        packaged=fullfile(root,'data','splits',[cfg.dataset '_splits.mat']);
        if isfile(packaged), split_file=packaged; end
    end
end
splits=repmat(struct('seed',0,'train_linear',[],'test_linear',[],'provenance',''),numel(seeds),1);
from_file=false(numel(seeds),1);
if isfile(split_file)
    saved=load(split_file);
    if isfield(saved,'splits'), stored=saved.splits;
    elseif isfield(saved,'S'), stored=saved.S;
    else, error('Release:Splits','Split file lacks S or splits: %s',split_file); end
    available=[stored.seed];
    for k=1:numel(seeds)
        i=find(available==seeds(k),1);
        if ~isempty(i), splits(k)=stored(i); from_file(k)=true; end
    end
    fprintf('Split file %s: %d/%d seeds reused\n',split_file,sum(from_file),numel(seeds));
end
missing=seeds(~from_file);
if ~isempty(missing)
    if isfile(split_file)
        fprintf('WARNING: regenerating splits for seeds %s (not in %s).\n', ...
            mat2str(missing),split_file);
    end
    gen=arrayfun(@(s)make_split(gt,cfg,s),missing,'UniformOutput',false);
    splits(~from_file)=[gen{:}];
end
validate_splits(gt,splits,seeds);
if ~isfolder(output_root), mkdir(output_root); end
run_dir=fullfile(output_root,sprintf('%s_%s',cfg.dataset,datestr(now,'yyyymmdd_HHMMSS_FFF')));
if isfolder(run_dir), run_dir=[run_dir '_2']; end
mkdir(run_dir);
save(fullfile(run_dir,'protocol.mat'),'cfg','segmentation','source','splits','-v7.3');
results=zeros(numel(seeds),7);
for k=1:numel(seeds)
    split=splits(k);
    result=run_pipeline(X,gt,split,cfg,state);
    result.segmentation=segmentation; result.source=source;
    result.matlab_version=version;
    if strcmpi(dataset,'demo'), result.reproduction_status='synthetic_demo_not_benchmark'; end
    raw=result.Raw_TDLPP.metrics; purified=result.SPPR_TDLPP.metrics;
    results(k,:)=[seeds(k),raw.OA,raw.AA,raw.Kappa, ...
        purified.OA,purified.AA,purified.Kappa];
    save(fullfile(run_dir,sprintf('seed_%d.mat',seeds(k))),'result','-v7.3');
    fprintf('%s seed %d: TDLPP %.4f%%, TDLPP+SPPR %.4f%%, gain %+.4f pp\n', ...
        cfg.dataset,seeds(k),100*raw.OA,100*purified.OA,100*(purified.OA-raw.OA));
    progress=array2table(results(1:k,:),'VariableNames', ...
        {'seed','TDLPP_OA','TDLPP_AA','TDLPP_Kappa','SPPR_OA','SPPR_AA','SPPR_Kappa'});
    progress.gain_pp=100*(progress.SPPR_OA-progress.TDLPP_OA);
    writetable(progress,fullfile(run_dir,'trials.csv'));
end
results=progress;
fprintf('Saved models, predictions, splits and metrics to %s\n',run_dir);
end

function cfg=configuration(dataset)
cfg=struct('version','minimal_unique_softclip_v2','dataset',char(dataset), ...
    'window',11,'dimension',50,'train_ratio',0.05,'minimum_train',10, ...
    'train_per_class',[],'test_per_class',[],'pca_dimension',0, ...
    'target_area',800,'ers_lambda',0.5,'ers_sigma',5,'batch_size',250, ...
    'centering','partition','max_iterations',50,'tolerance',0.005, ...
    'ridge',0,'t1',[],'t2',[],'classifier','frozen_fitcknn', ...
    'segmentation_source','recompute');
switch lower(char(dataset))
    case 'demo'
        cfg.window=3; cfg.dimension=2; cfg.train_ratio=0.2;
        cfg.minimum_train=1; cfg.batch_size=23;
    case {'indian','indian pines'}
        cfg.dataset='Indian'; cfg.folder='01_Indian_Pines';
        cfg.cube_file='Indian_pines_corrected.mat'; cfg.cube_variable='indian_pines_corrected';
        cfg.gt_file='Indian_pines_gt.mat'; cfg.gt_variable='indian_pines_gt';
    case 'salinas'
        cfg.dataset='Salinas'; cfg.folder='02_Salinas'; cfg.train_ratio=0.01;
        cfg.cube_file='Salinas_corrected.mat'; cfg.cube_variable='salinas_corrected';
        cfg.gt_file='Salinas_gt.mat'; cfg.gt_variable='salinas_gt';
    case {'honghu','whu-hi-honghu'}
        cfg.dataset='HongHu'; cfg.folder='WHU-Hi-HongHu'; cfg.dimension=20;
        cfg.train_per_class=50; cfg.test_per_class=300; cfg.pca_dimension=80;
        cfg.cube_file='WHU_Hi_HongHu.mat'; cfg.cube_variable='WHU_Hi_HongHu';
        cfg.gt_file='WHU_Hi_HongHu_gt.mat'; cfg.gt_variable='WHU_Hi_HongHu_gt';
    case {'houston','houston2013'}
        cfg.dataset='Houston'; cfg.folder='Houston'; cfg.dimension=15;
        cfg.cube_file='Houston.mat'; cfg.cube_variable='Houston';
        cfg.gt_file='Houston_gt.mat'; cfg.gt_variable='Houston_gt';
    otherwise
        error('Unknown dataset: %s',char(dataset));
end
end

function ensure_ers(root)
here=fullfile(root,'third_party','ers'); src=fullfile(here,'source');
if isfile(fullfile(here,['mex_ers.' mexext])), return; end
names={'mex_ers.cpp','MERCCInput.cpp','MERCOutput.cpp', ...
    'MERCDisjointSet.cpp','MERCFunctions.cpp','MERCLazyGreedy.cpp'};
files=cellfun(@(name)fullfile(src,name),names,'UniformOutput',false);
mex('-R2017b',['-I' src],'-outdir',here,'-output','mex_ers',files{:});
rehash;
end

function result = run_pipeline(X,gt,split,cfg,state)
%RUN_PIPELINE Paired Raw+TDLPP and frozen unique SoftClip+TDLPP.
% Test patches are streamed; only their feature matrix is retained.
if nargin<5, state=[]; end
assert(isequal(size(gt),size(X,[1 2])) && all(isfinite(X(:))));
assert(all(gt(:)>=0) && all(gt(:)==floor(gt(:))),'GT must contain nonnegative integer labels.');
train=double(split.train_linear(:)); test=double(split.test_linear(:));
validate_indices(train,numel(gt)); validate_indices(test,numel(gt));
assert(isempty(intersect(train,test)),'Training and testing centers overlap.');
assert(all(gt([train;test])>0),'Split contains unlabeled centers.');
assert(all(ismember(unique(gt(test)),unique(gt(train)))),'Test class absent from training.');
validateattributes(cfg.batch_size,{'numeric'},{'scalar','integer','positive'});
assert(ismember(cfg.centering,{'partition','training'}),'Unknown centering mode.');
assert(cfg.ridge>=0 && cfg.tolerance>=0 && cfg.max_iterations>=1);
% Caller provides the PREPROCESSED cube so external pseudo-maps cannot
% silently be applied to a different PCA/normalization representation.
segmentation=struct('provided_pseudo_map',~isempty(state),'ground_truth_used',false);
timer=tic;
if isempty(state), [state,segmentation]=sppr_segment(X,cfg); end
assert(isequal(size(state.pseudo_map),size(gt)) && state.height==size(X,1) && state.width==size(X,2));
segmentation.seconds=toc(timer);
[rows,cols]=ind2sub(size(gt),train); training_labels=double(gt(train)); truth=double(gt(test));
variants={'Raw_TDLPP','SPPR_TDLPP'};
result=struct('version',cfg.version,'config',cfg,'split',split, ...
    'segmentation',segmentation,'sppr_algorithm',state.algorithm, ...
    'reproduction_status','minimal_release_generated_splits_recomputed_ers');
for arm=1:2
    timer=tic; blocks=cell(numel(train),1);
    for k=1:numel(train)
        if arm==1, blocks{k}=extract_patch(X,rows(k),cols(k),cfg.window);
        else, blocks{k}=sppr_reconstruct(X,state,rows(k),cols(k),cfg.window); end
    end
    train_mean=zeros(size(blocks{1}),'single');
    for k=1:numel(blocks), train_mean=train_mean+single(blocks{k}); end
    train_mean=train_mean/numel(blocks);
    for k=1:numel(blocks), blocks{k}=single(blocks{k})-train_mean; end
    model=tdlpp_fit(blocks,training_labels,cfg);
    train_seconds=toc(timer); clear blocks
    timer=tic; [features,clcr]=test_features(X,gt,test,cfg,state,model,arm==2,train_mean);
    if strcmp(cfg.classifier,'frozen_fitcknn')
        classifier=fitcknn(model.train_features,training_labels(:), ...
            'NumNeighbors',1,'Distance','euclidean','Standardize',false);
        prediction=predict(classifier,features);
    else
        error('Unknown classifier: %s',cfg.classifier);
    end
    output=struct('model',model,'prediction',prediction,'truth',truth, ...
        'metrics',classification_metrics(prediction,truth,unique(training_labels)), ...
        'train_seconds',train_seconds,'test_seconds',toc(timer),'clcr',clcr);
    result.(variants{arm})=output;
end
result.OA_gain=result.SPPR_TDLPP.metrics.OA-result.Raw_TDLPP.metrics.OA;
end
function [features,diagnostics]=test_features(X,gt,indices,cfg,state,model,purify,train_mean)
[rows,cols]=ind2sub(size(gt),indices); n=numel(indices);
features=zeros(n,size(model.train_features,2)); total=zeros(size(train_mean),'single');
diagnostics=[];
for first=1:cfg.batch_size:n
    last=min(n,first+cfg.batch_size-1); blocks=cell(last-first+1,1);
    for k=first:last
        if purify
            [patch,info]=sppr_reconstruct(X,state,rows(k),cols(k),cfg.window);
        else
            patch=extract_patch(X,rows(k),cols(k),cfg.window);
        end
        total=total+single(patch); blocks{k-first+1}=patch;
    end
    features(first:last,:)=tdlpp_project(blocks,model);
end
if strcmp(cfg.centering,'partition'), offset=total/n; else, offset=train_mean; end
features=features-tdlpp_project({offset},model);
end
function validate_indices(indices,count)
assert(~isempty(indices) && all(indices>=1 & indices<=count & indices==floor(indices)), ...
    'Split must contain original-scene linear indices.');
assert(numel(unique(indices))==numel(indices),'Duplicate split indices.');
end

function validate_splits(gt,splits,seeds)
%VALIDATE_SPLITS Reject saved splits that do not describe this scene.
count=numel(gt);
assert(numel(splits)==numel(seeds),'Split count mismatch.');
for k=1:numel(splits)
    s=splits(k);
    assert(isequal(s.seed,seeds(k)),'Split seed mismatch at position %d.',k);
    fields={'train_linear','test_linear'};
    for f=1:2
        idx=double(s.(fields{f})(:));
        assert(~isempty(idx) && all(idx>=1 & idx<=count & idx==floor(idx)), ...
            'Split %d has out-of-range %s.',s.seed,fields{f});
        assert(numel(unique(idx))==numel(idx), ...
            'Split %d has duplicate %s.',s.seed,fields{f});
    end
    assert(isempty(intersect(s.train_linear(:),s.test_linear(:))), ...
        'Split %d overlaps train and test.',s.seed);
    assert(all(gt([s.train_linear(:);s.test_linear(:)])>0), ...
        'Split %d contains unlabeled centers.',s.seed);
end
end
function split = make_split(gt,cfg,seed)
%MAKE_SPLIT New validation splits, not the unpublished manuscript indices.
saved=rng; cleanup=onCleanup(@()rng(saved)); %#ok<NASGU>
rng(seed,'twister'); valid=find(gt>0); y=gt(valid); classes=unique(y);
train=[]; test=[];
for k=1:numel(classes)
    idx=find(y==classes(k)); order=idx(randperm(numel(idx)));
    if isempty(cfg.train_per_class)
        nt=max(cfg.minimum_train,floor(numel(idx)*cfg.train_ratio)); nv=numel(idx)-nt;
    else
        nt=cfg.train_per_class; nv=cfg.test_per_class;
    end
    assert(nt>=1 && nv>=1 && nt+nv<=numel(idx), ...
        'Class %g has insufficient samples for requested partition.',classes(k));
    train=[train;order(1:nt)]; test=[test;order(nt+1:nt+nv)]; %#ok<AGROW>
end
split=struct('seed',seed,'train_linear',valid(train),'test_linear',valid(sort(test)), ...
    'provenance','new_public_validation_split_not_paper_original');
end

function metrics = classification_metrics(predicted,truth,classes)
%CLASSIFICATION_METRICS OA, AA and kappa in [0,1], arbitrary class IDs.
predicted=predicted(:); truth=truth(:);
assert(numel(predicted)==numel(truth) && ~isempty(truth));
if nargin<3, classes=unique(truth); end
[ok1,a]=ismember(predicted,classes); [ok2,b]=ismember(truth,classes);
assert(all(ok1 & ok2),'Labels outside the specified class set.');
C=numel(classes); CM=accumarray([b,a],1,[C,C]); total=sum(CM,'all');
counts=sum(CM,2); acc=diag(CM)./counts; oa=trace(CM)/total;
pe=sum(sum(CM,1)'.*counts)/total^2;
if pe==1, kappa=NaN; else, kappa=(oa-pe)/(1-pe); end
metrics=struct('OA',oa,'AA',mean(acc(counts>0)),'Kappa',kappa, ...
    'classes',classes(:),'per_class_accuracy',acc,'confusion',CM, ...
    'unit','proportion','confusion_rows','true_labels');
end
