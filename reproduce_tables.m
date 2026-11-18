function summary = reproduce_tables(data_root,output_root,seeds)
%REPRODUCE_TABLES Paired TDLPP and SPPR+TDLPP runs on all four datasets.
% summary = reproduce_tables;                        % data/, results/, 1001:1010
% summary = reproduce_tables('D:/hsi','out');        % custom roots
% summary = reproduce_tables([],[],[1001 1002]);     % two splits, quick check
%
% Writes trials_all.csv (per split) and table10_like.csv (mean/std per
% dataset and method) under a new summary folder in output_root.
root=fileparts(mfilename('fullpath'));
if nargin<1 || isempty(data_root), data_root=fullfile(root,'data'); end
if nargin<2 || isempty(output_root), output_root=fullfile(root,'results'); end
if nargin<3 || isempty(seeds), seeds=1001:1010; end
datasets={'Indian','Salinas','HongHu','Houston'};
rows=cell(0,7);
for d=1:numel(datasets)
    [t,run_dir]=main(datasets{d},data_root,output_root,seeds);
    for k=1:height(t)
        rows(end+1,:)={datasets{d},'TDLPP',t.seed(k),100*t.TDLPP_OA(k), ...
            100*t.TDLPP_AA(k),100*t.TDLPP_Kappa(k),run_dir}; %#ok<AGROW>
        rows(end+1,:)={datasets{d},'TDLPP+SPPR',t.seed(k),100*t.SPPR_OA(k), ...
            100*t.SPPR_AA(k),100*t.SPPR_Kappa(k),run_dir}; %#ok<AGROW>
    end
end
all=cell2table(rows,'VariableNames', ...
    {'dataset','method','seed','OA','AA','Kappa','run_dir'});
out_dir=fullfile(output_root,sprintf('summary_%s',datestr(now,'yyyymmdd_HHMMSS')));
if ~isfolder(out_dir), mkdir(out_dir); end
writetable(all,fullfile(out_dir,'trials_all.csv'));

methods={'TDLPP','TDLPP+SPPR'}; agg=cell(0,9);
for d=1:numel(datasets)
    for m=1:2
        s=all(strcmp(all.dataset,datasets{d}) & strcmp(all.method,methods{m}),:);
        agg(end+1,:)={datasets{d},methods{m},height(s),mean(s.OA),std(s.OA), ...
            mean(s.AA),std(s.AA),mean(s.Kappa),std(s.Kappa)}; %#ok<AGROW>
    end
end
summary=cell2table(agg,'VariableNames',{'dataset','method','n', ...
    'OA_mean','OA_std','AA_mean','AA_std','Kappa_mean','Kappa_std'});
% SPPR rows are even; gain is reported once per dataset on the SPPR row.
summary.gain_pp=nan(height(summary),1);
summary.gain_pp(2:2:end)=summary.OA_mean(2:2:end)-summary.OA_mean(1:2:end);
writetable(summary,fullfile(out_dir,'table10_like.csv'));

fprintf('\n%-9s %-11s %3s %15s %15s %15s\n','dataset','method','n', ...
    'OA','AA','Kappa');
for r=1:height(summary)
    fprintf('%-9s %-11s %3d %7.2f +-%5.2f %7.2f +-%5.2f %7.2f +-%5.2f\n', ...
        summary.dataset{r},summary.method{r},summary.n(r), ...
        summary.OA_mean(r),summary.OA_std(r),summary.AA_mean(r), ...
        summary.AA_std(r),summary.Kappa_mean(r),summary.Kappa_std(r));
end
fprintf('\nSPPR gain (pp):');
g=summary.gain_pp(2:2:end);
for d=1:numel(datasets), fprintf(' %s %+.2f |',datasets{d},g(d)); end
fprintf('\nSaved %s\n',fullfile(out_dir,'table10_like.csv'));
end
