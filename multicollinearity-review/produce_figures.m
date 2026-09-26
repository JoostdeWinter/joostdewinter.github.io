%% Multicollinearity in Statistical Modeling: Replication Script
% Reproduces Figures 1–4 and the simulation results in Textboxes 1–4 of:
%   De Winter, J. C. F. (2026). Multicollinearity in statistical modeling:
%   A review. https://joostdewinter.github.io/multicollinearity-review.pdf
%
% Output: four 600-dpi PNG figures, CSV tables, and replication_results.mat,
% saved in the folder multicollinearity_outputs.
%
% Requirements: MATLAB R2020a or later with the Statistics and Machine
% Learning Toolbox. The Parallel Computing Toolbox is optional; without it,
% the Monte Carlo analysis runs serially.
%
% Local helper functions are defined at the end of the script.

clear; close all; clc;

OUTPUT_DIR = fullfile(pwd, 'multicollinearity_outputs');
if ~exist(OUTPUT_DIR, 'dir')
    mkdir(OUTPUT_DIR);
end
EXPORT_DPI = 600;

% Figure 3 simulation settings.
% Every Monte Carlo replication, including the representative one shown in
% Figure 3, fits/tunes the models on n = 80 training observations and
% evaluates them on an independent N_TEST = 10,000 test sample.
N_MC = 1000;
N_TEST = 10000;
K_FOLDS = 10;
RIDGE_GRID = logspace(-4,3,60);
LASSO_NUM_LAMBDA = 200;
LASSO_LAMBDA_RATIO = 1e-6;
USE_PARALLEL = true;

% Preferred figure window positions and sizes [left bottom width height].
% These values were captured after manual resizing in MATLAB.
FIG1_POSITION = [248 128 1328 720];
FIG2_POSITION = [451 428 1223 711];
FIG3_POSITION = [60 60 1750 930];
FIG4_POSITION = [120 446 1209 793];

% Figure 1 axis-label placement.
% For 3-D axes, explicit data-coordinate labels can collide with tick labels
% because of perspective projection. Instead, x1 and x2 are placed in
% normalized axes coordinates. These positions are stable for the fixed view.
%
% Tune only these two [x y] positions if desired.
FIG1_XLABEL_POSITION = [0.60 -0.035];
FIG1_YLABEL_POSITION = [0.025 0.120];

%% ========================================================================
%  SECTION 1: FIGURE 1 & TEXTBOX 1
%  Illustration of Multicollinearity
%  ========================================================================
fprintf('<strong>--- Generating Figure 1 & Textbox 1 Data ---</strong>\n');

rng(1);
fig1 = figure('Name','Figure 1: 3D Illustration', ...
    'Color','w','Position',FIG1_POSITION);

% Compact side-by-side layout. A single shared colorbar is added after
% both panels, which avoids wasting space between the panels.
tl1 = tiledlayout(fig1,1,2,'Padding','compact','TileSpacing','compact');

fig1_r12 = zeros(2,1);
fig1_b1_pop = zeros(2,1);
fig1_b2_pop = zeros(2,1);
fig1_b1_est = zeros(2,1);
fig1_b2_est = zeros(2,1);
fig1_R2_pop = zeros(2,1);
fig1_R2_est = zeros(2,1);
fig1_VIF = zeros(2,1);

for fg = 1:2
    if fg == 1
        R_12 = 0;
    else
        R_12 = 0.9;
    end

    ax = nexttile(tl1,fg);

    % Population correlation matrix.
    r_mat = [1 R_12 0.5; R_12 1 0.4; 0.5 0.4 1];
    Rxx = r_mat(1:2,1:2);
    rxy = r_mat(1:2,3);

    % Population multiple-regression coefficients and diagnostics.
    beta_pop = Rxx \ rxy;
    R2_pop = rxy' * beta_pop;
    vif_pop = diag(inv(Rxx));

    % Generate data (n = 10,000).
    L = chol(r_mat);
    X = randn(10000,3) * L;
    Xz = (X - mean(X)) ./ std(X);
    x1 = Xz(:,1);
    x2 = Xz(:,2);
    y  = Xz(:,3);

    % OLS fit.
    st = regstats(y,[x1 x2],'linear',{'beta','rsquare'});
    b0 = st.beta(1);
    b1 = st.beta(2);
    b2 = st.beta(3);

    % Plot.
    scatter3(ax,x1,x2,y,20,y,'filled');
    hold(ax,'on');

    x1fit = min(x1):0.2:max(x1);
    x2fit = min(x2):0.2:max(x2);
    [X1FIT,X2FIT] = meshgrid(x1fit,x2fit);
    YFIT = b0 + b1*X1FIT + b2*X2FIT;
    surf(ax,X1FIT,X2FIT,YFIT, ...
        'FaceColor','k','EdgeColor','k','FaceAlpha',0.2);

    % Both panels use the same color scale, allowing one shared colorbar.
    caxis(ax,[-4 4]);

    % Suppress MATLAB's automatic x/y labels. Custom labels are added
    % below in normalized axes coordinates.
    xlabel(ax,'');
    ylabel(ax,'');
    zlabel(ax,'\it{y}\rm');
    title(ax,sprintf('\\it{r}\\rm_{12,\\it{p}\\rm} = %.2f  (VIF = %.2f)', ...
        R_12,vif_pop(1)),'FontWeight','normal');

    box(ax,'on');
    grid(ax,'on');
    view(ax,-22,32);
    light(ax,'Position',[1 0 1],'Style','infinite');
    lighting(ax,'gouraud');
    material(ax,'shiny');
    xlim(ax,[-4 4]); ylim(ax,[-4 4]); zlim(ax,[-4 4]);

    % Place x1 and x2 in normalized axes coordinates. This avoids the
    % perspective-induced collisions that occurred with data-coordinate text.
    hx = text(ax,FIG1_XLABEL_POSITION(1),FIG1_XLABEL_POSITION(2), ...
        '\it{x}\rm_1', ...
        'Units','normalized','Interpreter','tex', ...
        'FontName','Arial','FontSize',20, ...
        'HorizontalAlignment','center','VerticalAlignment','middle', ...
        'Clipping','off');

    hy = text(ax,FIG1_YLABEL_POSITION(1),FIG1_YLABEL_POSITION(2), ...
        '\it{x}\rm_2', ...
        'Units','normalized','Interpreter','tex', ...
        'FontName','Arial','FontSize',20, ...
        'HorizontalAlignment','center','VerticalAlignment','middle', ...
        'Clipping','off');

    % Retain the v3 Figure 1 font size.
    set(findall(ax,'-property','FontName'),'FontName','Arial');
    set(findall(ax,'-property','FontSize'),'FontSize',20);
    set(ax,'LooseInset',[0.01 0.01 0.01 0.01]);

    % Console output.
    fprintf('\nScenario: r12_p = %.2f\n',R_12);
    fprintf('  Population beta 1: %.5f | Estimated: %.5f\n',beta_pop(1),b1);
    fprintf('  Population beta 2: %.5f | Estimated: %.5f\n',beta_pop(2),b2);
    fprintf('  Population R^2:    %.5f | Estimated: %.5f\n',R2_pop,st.rsquare);
    fprintf('  Population VIF:    %.5f\n',vif_pop(1));

    % Store.
    fig1_r12(fg) = R_12;
    fig1_b1_pop(fg) = beta_pop(1);
    fig1_b2_pop(fg) = beta_pop(2);
    fig1_b1_est(fg) = b1;
    fig1_b2_est(fg) = b2;
    fig1_R2_pop(fg) = R2_pop;
    fig1_R2_est(fg) = st.rsquare;
    fig1_VIF(fg) = vif_pop(1);
end

% One shared colorbar for both panels.
cb1 = colorbar(ax);
cb1.Layout.Tile = 'east';
cb1.FontName = 'Arial';
cb1.FontSize = 20;
cb1.Label.String = '\it{y}';
cb1.Label.FontName = 'Arial';
cb1.Label.FontSize = 20;

Figure1_results = table(fig1_r12,fig1_b1_pop,fig1_b2_pop, ...
    fig1_b1_est,fig1_b2_est,fig1_R2_pop,fig1_R2_est,fig1_VIF, ...
    'VariableNames',{'r12_population','beta1_population','beta2_population', ...
    'beta1_estimated','beta2_estimated','R2_population','R2_estimated','VIF'});
writetable(Figure1_results,fullfile(OUTPUT_DIR,'Figure1_results.csv'));

drawnow;
exportgraphics(fig1,fullfile(OUTPUT_DIR,'Figure1.png'), ...
    'Resolution',EXPORT_DPI,'BackgroundColor','white');

%% ========================================================================
%  SECTION 2: FIGURE 2 & TEXTBOX 2
%  Sampling Error Amplification ("Bouncing Betas")
%
%  The true regression coefficients and residual SD are held fixed across
%  the two conditions. Thus, the main design change is predictor correlation.
%  ========================================================================
fprintf('\n<strong>--- Generating Figure 2 & Textbox 2 Data ---</strong>\n');

tic;
reps = 10000;
n = 50;
R12_vals = [0 0.80];
beta_true_fig2 = [0.40; 0.40];
eps_sd_fig2 = 1.0;

Bstore = nan(numel(R12_vals),reps,2);

for ii = 1:numel(R12_vals)
    rng(ii*6,'twister');
    r12p = R12_vals(ii);
    Sigma2 = [1 r12p; r12p 1];
    L = chol(Sigma2);

    for i = 1:reps
        X = randn(n,2) * L;
        y = X*beta_true_fig2 + eps_sd_fig2*randn(n,1);
        D = [ones(n,1) X];
        b = D \ y;
        Bstore(ii,i,:) = b(2:3);
    end
end
toc;

mean_b1 = squeeze(mean(Bstore(:,:,1),2));
mean_b2 = squeeze(mean(Bstore(:,:,2),2));
sd_b1   = squeeze(std(Bstore(:,:,1),[],2));
sd_b2   = squeeze(std(Bstore(:,:,2),[],2));
corr_b1b2 = zeros(numel(R12_vals),1);
VIF_fig2 = 1 ./ (1 - R12_vals(:).^2);

for ii = 1:numel(R12_vals)
    C = corrcoef(squeeze(Bstore(ii,:,1)),squeeze(Bstore(ii,:,2)));
    corr_b1b2(ii) = C(1,2);
end

Figure2_results = table(R12_vals(:),VIF_fig2,mean_b1,mean_b2, ...
    sd_b1,sd_b2,corr_b1b2, ...
    'VariableNames',{'r12_population','VIF','Mean_beta1','Mean_beta2', ...
    'SD_beta1','SD_beta2','Corr_beta1_beta2'});
writetable(Figure2_results,fullfile(OUTPUT_DIR,'Figure2_results.csv'));

fprintf('\nFigure 2 sampling-distribution summary:\n');
disp(Figure2_results);

% Common limits.
x0  = squeeze(Bstore(1,:,1));
y0  = squeeze(Bstore(1,:,2));
x08 = squeeze(Bstore(2,:,1));
y08 = squeeze(Bstore(2,:,2));

allvals = [x0(:); y0(:); x08(:); y08(:)];
lo = min(allvals);
hi = max(allvals);
pad = 0.04*(hi-lo);
plotlims = [lo-pad hi+pad];

% Retain v3 Figure 2 size and 30-point font scale.
fig2 = figure('Color','w','Name','Figure 2: Bouncing Betas', ...
    'Position',FIG2_POSITION);

% Manual axes positions retain the large fonts while giving the left
% y-label enough margin and leaving a clearer gap between the panels.
ax1 = axes(fig2,'Position',[0.115 0.14 0.35 0.74]);
hold(ax1,'on');
scatter(ax1,x0,y0,20,'filled', ...
    'MarkerFaceColor',[.5 .5 .5], ...
    'MarkerEdgeColor','none','MarkerFaceAlpha',0.10);
xline(ax1,beta_true_fig2(1),'k:','LineWidth',2);
yline(ax1,beta_true_fig2(2),'k:','LineWidth',2);
plot(ax1,beta_true_fig2(1),beta_true_fig2(2),'kx', ...
    'MarkerSize',18,'LineWidth',4);
axis(ax1,'equal');
xlim(ax1,plotlims); ylim(ax1,plotlims);
xlabel(ax1,'\beta_1'); ylabel(ax1,'\beta_2');
title(ax1,sprintf('\\it{r}\\rm_{12,\\it{p}\\rm} = %.2f  (VIF = %.2f)', ...
    R12_vals(1),VIF_fig2(1)), ...
    'FontWeight','normal');
grid(ax1,'on'); box(ax1,'on');

ax2 = axes(fig2,'Position',[0.585 0.14 0.35 0.74]);
hold(ax2,'on');
scatter(ax2,x08,y08,20,'filled', ...
    'MarkerFaceColor',[.5 .5 .5], ...
    'MarkerEdgeColor','none','MarkerFaceAlpha',0.10);
xline(ax2,beta_true_fig2(1),'k:','LineWidth',2);
yline(ax2,beta_true_fig2(2),'k:','LineWidth',2);
plot(ax2,beta_true_fig2(1),beta_true_fig2(2),'kx', ...
    'MarkerSize',18,'LineWidth',4);
axis(ax2,'equal');
xlim(ax2,plotlims); ylim(ax2,plotlims);
xlabel(ax2,'\beta_1');
title(ax2,sprintf('\\it{r}\\rm_{12,\\it{p}\\rm} = %.2f  (VIF = %.2f)', ...
    R12_vals(2),VIF_fig2(2)), ...
    'FontWeight','normal');
grid(ax2,'on'); box(ax2,'on');

set(findall(fig2,'-property','FontName'),'FontName','Arial');
set(findall(fig2,'-property','FontSize'),'FontSize',30);
ax1.Title.FontSize = 27;
ax2.Title.FontSize = 27;

drawnow;
exportgraphics(fig2,fullfile(OUTPUT_DIR,'Figure2.png'), ...
    'Resolution',EXPORT_DPI,'BackgroundColor','white');

%% ========================================================================
%  SECTION 3: FIGURE 3 & TEXTBOX 3
%  Remedies: OLS vs Ridge vs Lasso
%  ========================================================================
fprintf('\n<strong>--- Generating Figure 3 & Textbox 3 Data ---</strong>\n');

n = 80;
p = 20;
rho = 0.9;
eps_sd = 4.0;
beta_true = [1.0;0;0.8;0.6;0;0;0.9;0;0.5;0;zeros(10,1)];

Sigma = eye(p);
Sigma(1:10,1:10) = rho;
Sigma(11:20,11:20) = rho;
Sigma(1:p+1:end) = 1;

% ------------------------------------------------------------------------
% Monte Carlo analysis.
%
% Each replication (figure3_replication_func, seed 10000 + replication):
%   1) generates a fresh n = 80 training sample;
%   2) tunes Ridge (minimum CV error) and Lasso (1-SE rule) by 10-fold CV
%      within that training sample; the Lasso solution at the minimum CV
%      error is kept as a countercheck;
%   3) evaluates OLS, Ridge, and Lasso on a fresh independent test sample
%      of N_TEST = 10,000 observations;
%   4) records test RMSE, coefficient MSE, and Lasso sparsity.
%
% The large independent test set removes most test-set evaluation noise while
% retaining the small-sample estimation problem in the n = 80 training data.
fprintf('\nRunning Figure 3 Monte Carlo analysis (%d replications; independent test N = %d)...\n', ...
    N_MC, N_TEST);

useParallelNow = false;
if USE_PARALLEL
    try
        if license('test','Distrib_Computing_Toolbox')
            pool = gcp('nocreate');
            if isempty(pool)
                fprintf('Starting a parallel pool for the Monte Carlo analysis...\n');
                pool = parpool;
            end
            useParallelNow = true;
            fprintf('Using %d parallel workers.\n',pool.NumWorkers);
        end
    catch ME
        warning('Parallel execution unavailable (%s). Continuing serially.',ME.message);
        useParallelNow = false;
    end
end
if ~useParallelNow
    fprintf('Using serial Monte Carlo execution.\n');
end

MC = figure3_run_mc_func(10000,N_MC,useParallelNow,n,p,Sigma,beta_true, ...
    eps_sd,N_TEST,RIDGE_GRID,K_FOLDS,LASSO_NUM_LAMBDA,LASSO_LAMBDA_RATIO);
rmseAll = MC.rmse;
coefMSEAll = MC.coef_mse;
lassoNnz = MC.lasso_nnz;

[~,winners] = min(rmseAll,[],2);
winCount = accumarray(winners,1,[3 1])';

methodNames = ["OLS";"Ridge";"Lasso"];
Figure3_MonteCarlo_results = table( ...
    methodNames, ...
    mean(rmseAll,1)', ...
    std(rmseAll,[],1)', ...
    prctile(rmseAll,2.5,1)', ...
    prctile(rmseAll,97.5,1)', ...
    100*winCount'/N_MC, ...
    mean(coefMSEAll,1)', ...
    std(coefMSEAll,[],1)', ...
    'VariableNames',{'Method','Mean_Test_RMSE','SD_Test_RMSE', ...
    'P2_5_Test_RMSE','P97_5_Test_RMSE','Lowest_RMSE_Percent', ...
    'Mean_Coefficient_MSE','SD_Coefficient_MSE'});

fprintf('\nFigure 3 Monte Carlo summary:\n');
disp(Figure3_MonteCarlo_results);
fprintf('Lasso number of nonzero coefficients: median = %.0f, range = %d-%d\n', ...
    median(lassoNnz),min(lassoNnz),max(lassoNnz));

writetable(Figure3_MonteCarlo_results, ...
    fullfile(OUTPUT_DIR,'Figure3_MonteCarlo_results.csv'));

writetable(table((1:N_MC)',lassoNnz, ...
    'VariableNames',{'Replication','Lasso_Nonzero_Count'}), ...
    fullfile(OUTPUT_DIR,'Figure3_Lasso_nonzero_counts.csv'));

% ---------------- Counterchecks reported in Textbox 3 ----------------
% (a) Lasso tuned at the minimum CV error (same fits as above).
% (b) Coefficient MSE of an all-zero estimate, as a reference value.
% (c) The same design with uncorrelated predictors (Sigma = identity),
%     using seeds 20000 + replication. With normally distributed predictors,
%     the expected OLS test MSE is eps_sd^2*(n+1)*(n-2)/(n*(n-p-2)),
%     whatever the predictor correlations.
fprintf('\nRunning Figure 3 countercheck with uncorrelated predictors (%d replications)...\n',N_MC);
MC0 = figure3_run_mc_func(20000,N_MC,useParallelNow,n,p,eye(p),beta_true, ...
    eps_sd,N_TEST,RIDGE_GRID,K_FOLDS,LASSO_NUM_LAMBDA,LASSO_LAMBDA_RATIO);

[~,winnersMin] = min([rmseAll(:,1:2) MC.rmse_lasso_min],[],2);
Quantity = [ ...
    "Lasso (minimum CV error): mean test RMSE"; ...
    "Lasso (minimum CV error): median number of nonzero coefficients"; ...
    "Ridge lowest test RMSE (%) against OLS and Lasso (minimum CV error)"; ...
    "All-zero estimate: mean coefficient MSE (standardized scale)"; ...
    "Uncorrelated predictors: OLS mean test RMSE"; ...
    "Uncorrelated predictors: Ridge mean test RMSE"; ...
    "Uncorrelated predictors: Lasso (1-SE rule) mean test RMSE"; ...
    "Square root of expected OLS test MSE, normal predictors (theory)"; ...
    "OLS coefficient MSE, original scale, correlated predictors"; ...
    "OLS coefficient MSE, original scale, uncorrelated predictors"; ...
    "Ratio of the two OLS coefficient MSEs"; ...
    "Within-block VIF (population)"];
Value = [ ...
    mean(MC.rmse_lasso_min); ...
    median(MC.lasso_min_nnz); ...
    100*mean(winnersMin == 2); ...
    mean(MC.coef_mse_zero); ...
    mean(MC0.rmse(:,1)); ...
    mean(MC0.rmse(:,2)); ...
    mean(MC0.rmse(:,3)); ...
    eps_sd*sqrt((n+1)*(n-2)/(n*(n-p-2))); ...
    mean(MC.coef_mse_raw(:,1)); ...
    mean(MC0.coef_mse_raw(:,1)); ...
    mean(MC.coef_mse_raw(:,1))/mean(MC0.coef_mse_raw(:,1)); ...
    max(diag(inv(Sigma)))];
Figure3_counterchecks = table(Quantity,Value);
fprintf('\nFigure 3 counterchecks:\n');
disp(Figure3_counterchecks);
writetable(Figure3_counterchecks,fullfile(OUTPUT_DIR,'Figure3_counterchecks.csv'));

% Select a representative realization for the LEFT Figure 3 panel.
% Criterion:
%   1) Lasso must select the median number of predictors.
%   2) Among those replications, choose the one whose test-RMSE vector is
%      closest to the Monte Carlo mean test-RMSE vector, standardized by the
%      Monte Carlo SDs. A small coefficient-MSE penalty is included only as a
%      mild tie-breaker.
target_nnz = round(median(lassoNnz));
cand = find(lassoNnz == target_nnz);

rmse_mean_all = mean(rmseAll,1);
rmse_sd_all = std(rmseAll,[],1);
rmse_sd_all(rmse_sd_all == 0) = 1;

coefmse_mean_all = mean(coefMSEAll,1);
coefmse_sd_all = std(coefMSEAll,[],1);
coefmse_sd_all(coefmse_sd_all == 0) = 1;

repScore = sum(((rmseAll(cand,:) - rmse_mean_all) ./ rmse_sd_all).^2, 2) + ...
           0.25 * sum(((coefMSEAll(cand,:) - coefmse_mean_all) ./ coefmse_sd_all).^2, 2);

[~,bestCandIdx] = min(repScore);
representative_replication = cand(bestCandIdx);
representative_seed = 10000 + representative_replication;

% Re-run the selected replication with the same seed to obtain its details.
Rrep = figure3_replication_func(representative_seed,n,p,Sigma,beta_true, ...
    eps_sd,N_TEST,RIDGE_GRID,K_FOLDS,LASSO_NUM_LAMBDA,LASSO_LAMBDA_RATIO);
b_ols = Rrep.bo;
b_rdg = Rrep.br;
b_las = Rrep.bl;
b0_rdg = Rrep.b0r;
b0_las = Rrep.b0l;
k_best = Rrep.ks;
rmse_te = Rrep.rmse;
r2_te = Rrep.r2;
beta_true_std = Rrep.btrue_s;
representative_lasso_predictors = Rrep.lasso_idx;

fprintf('\nRepresentative Figure 3 replication selected automatically:\n');
fprintf('  Replication index = %d (seed = %d)\n', ...
    representative_replication, representative_seed);
fprintf('  Lasso selected %d predictors (target median = %d): %s\n', ...
    numel(representative_lasso_predictors), target_nnz, ...
    mat2str(representative_lasso_predictors(:)'));
fprintf('  Test RMSE: OLS=%.2f | Ridge(k*=%.3g)=%.2f | Lasso=%.2f\n', ...
    rmse_te(1), k_best, rmse_te(2), rmse_te(3));
fprintf('  Test R^2: OLS=%.2f | Ridge=%.2f | Lasso=%.2f\n', r2_te);
fprintf('  Representative-score = %.3f (smaller = more typical)\n', ...
    repScore(bestCandIdx));


% ---------------- Figure 3 plotting ----------------
% Retain v3 dimensions and font scale:
% axes 24, general text 26, legend 23.
cBase = [0.20 0.40 0.85; ...
         0.90 0.50 0.10; ...
         0.20 0.65 0.35];
fig3 = figure('Color','w','Name','Figure 3: Remedies', ...
    'Position',FIG3_POSITION);
tl3 = tiledlayout(fig3,1,2,'Padding','loose','TileSpacing','loose');

% Left: coefficients.
ax31 = nexttile(tl3);
hold(ax31,'on'); box(ax31,'on'); grid(ax31,'on');

Bcoef = [b_ols,b_rdg,b_las];
Bcoef(abs(Bcoef)<1e-3) = 0;

hb = bar(ax31,Bcoef,'LineWidth',0.5,'BarWidth',0.88);
for i = 1:3
    set(hb(i),'FaceColor',cBase(i,:),'EdgeColor','none');
end

plot(ax31,1:p,beta_true_std,'ko', ...
    'MarkerFaceColor','w','MarkerSize',6,'LineWidth',1.8);
yline(ax31,0,'k:');
xline(ax31,10.5,'--','Color',[0.2 0.2 0.2], ...
    'LineWidth',2.0,'Alpha',0.8);

xlabel(ax31,'Predictor \it{x}\rm');
ylabel(ax31,'\beta');
title(ax31,'Representative realization: coefficient estimates','FontWeight','normal');
set(ax31,'XTick',1:p,'FontName','Arial','FontSize',24);

% Inside legend, two columns, avoids competing with title.
lg31 = legend(ax31,{'OLS','Ridge','Lasso','True \beta'}, ...
    'Location','northwest','NumColumns',2,'Box','off');
lg31.FontName = 'Arial';
lg31.FontSize = 23;

mB = max(abs([Bcoef(:);beta_true_std(:)]));
ylim(ax31,1.12*mB*[-1 1]);

% Right: Monte Carlo mean test RMSE with 95% simulation intervals.
ax32 = nexttile(tl3);
hold(ax32,'on'); box(ax32,'on'); grid(ax32,'on');

cats = reordercats(categorical({'OLS','Ridge','Lasso'}), ...
    {'OLS','Ridge','Lasso'});

mc_mean = Figure3_MonteCarlo_results.Mean_Test_RMSE;
mc_lo = Figure3_MonteCarlo_results.P2_5_Test_RMSE;
mc_hi = Figure3_MonteCarlo_results.P97_5_Test_RMSE;
mc_err_low = mc_mean - mc_lo;
mc_err_high = mc_hi - mc_mean;

bh = bar(ax32,cats,mc_mean(:), ...
    'LineWidth',0.5,'BarWidth',0.62,'FaceColor','flat');
bh.CData = cBase;
bh.EdgeColor = 'none';

er = errorbar(ax32,bh.XEndPoints,mc_mean(:),mc_err_low(:),mc_err_high(:), ...
    'k','LineStyle','none','LineWidth',1.8,'CapSize',10);

ylabel(ax32,'Mean test RMSE (original \it{y}\rm-units)');
% Title with the number of replications, using a thousands separator.
mcLabel = regexprep(sprintf('%d',N_MC),'(\d)(?=(\d{3})+$)','$1,');
title(ax32,['Mean test RMSE (' mcLabel ' replications)'],'FontWeight','normal');

maxRMSE = max(mc_hi);
ylim(ax32,[0 1.22*maxRMSE]);

% Place the numeric labels above the tops of the 95% interval bars.
labelY = mc_hi(:) + 0.025*maxRMSE;
text(ax32,bh.XEndPoints,labelY, ...
    compose('%.2f',bh.YData), ...
    'HorizontalAlignment','center', ...
    'VerticalAlignment','bottom', ...
    'FontWeight','bold','FontName','Arial','FontSize',22);

set(ax32,'FontName','Arial','FontSize',24);
set(findall(fig3,'-property','FontName'),'FontName','Arial');

drawnow;
exportgraphics(fig3,fullfile(OUTPUT_DIR,'Figure3.png'), ...
    'Resolution',EXPORT_DPI,'BackgroundColor','white');

%% ========================================================================
%  SECTION 4: FIGURE 4 & TEXTBOX 4
%  Moderated Regression & Mean Centering
%  ========================================================================
fprintf('\n<strong>--- Generating Figure 4 & Textbox 4 Data ---</strong>\n');

rng(2025,'twister');

n = 800;
rho = 0.50;
muX = 2;
muZ = -1;
beta = [0;1;1;1.5];
eps_sd = 1.0;

Sigma4 = [1 rho; rho 1];
XZ = mvnrnd([muX muZ],Sigma4,n);
X = XZ(:,1);
Z = XZ(:,2);

y = beta(1) + beta(2)*X + beta(3)*Z + ...
    beta(4)*(X.*Z) + eps_sd*randn(n,1);

meanX = mean(X);
meanZ = mean(Z);

Xc = X - meanX;
Zc = Z - meanZ;

D1 = [ones(n,1) X  Z  X.*Z];
D2 = [ones(n,1) Xc Zc Xc.*Zc];

[b1,se1,~,~,~,V1] = ols_with_se_func(D1,y);
[b2,se2,~,~,~,V2] = ols_with_se_func(D2,y);

kappa1 = bkw_condition_number_func(D1);
kappa2 = bkw_condition_number_func(D2);

vif1 = vif_func(D1(:,2:end));
vif2 = vif_func(D2(:,2:end));

% Verify identical fitted model.
yhat1 = D1*b1;
yhat2 = D2*b2;
maxFitDiff = max(abs(yhat1-yhat2));
SSE1 = sum((y-yhat1).^2);
SSE2 = sum((y-yhat2).^2);
R2_1 = 1 - SSE1/sum((y-mean(y)).^2);
R2_2 = 1 - SSE2/sum((y-mean(y)).^2);

% The same quantities computed from both parameterizations.
% 1) slope of X at original Z = 0
% 2) slope of X at original Z = mean(Z)
% 3) interaction
C_unc = [0 1 0 0; ...
         0 1 0 meanZ; ...
         0 0 0 1];

C_cent = [0 1 0 -meanZ; ...
          0 1 0 0; ...
          0 0 0 1];

est_unc = C_unc*b1;
est_cent = C_cent*b2;

se_unc = sqrt(diag(C_unc*V1*C_unc'));
se_cent = sqrt(diag(C_cent*V2*C_cent'));

fprintf('Interaction Model Diagnostics:\n');
% Here the maximum Belsley condition index equals the singular-value
% condition number of the column-scaled design matrix. The term "condition
% number" is used to avoid confusing CI with confidence interval.
fprintf('  Model 1 (uncentered): Condition number = %.2f\n',kappa1);
fprintf('  Model 2 (centered):   Condition number = %.2f\n',kappa2);
fprintf('  Max fitted-value difference = %.3e\n',maxFitDiff);
fprintf('  SSE difference = %.3e\n',abs(SSE1-SSE2));
fprintf('  R^2 difference = %.3e\n',abs(R2_1-R2_2));
fprintf('  Interaction coefficient: Unc.=%.5f, Cent.=%.5f\n',b1(4),b2(4));
fprintf('  VIF Unc.  [x z xz] = [%.2f %.2f %.2f]\n',vif1);
fprintf('  VIF Cent. [x z xz] = [%.2f %.2f %.2f]\n',vif2);

Figure4_equivalent_estimands = table( ...
    ["Slope at z=0";"Slope at mean(z)";"Interaction"], ...
    est_unc,est_cent,se_unc,se_cent, ...
    'VariableNames',{'Estimand','Uncentered_Estimate','Centered_Estimate', ...
    'Uncentered_SE','Centered_SE'});
writetable(Figure4_equivalent_estimands, ...
    fullfile(OUTPUT_DIR,'Figure4_equivalent_estimands.csv'));

% ---------------- Figure 4 plotting ----------------
% Retain v3 dimensions and font scale:
% axes 22, general text 24, legend 21.
cUn = [0.20 0.40 0.85];
cCe = [0.90 0.50 0.10];

fig4 = figure('Color','w','Name','Figure 4: Mean Centering', ...
    'Position',FIG4_POSITION);
tl4 = tiledlayout(fig4,1,2,'Padding','loose','TileSpacing','loose');

% Reserve extra space below the panels for the diagonal x-axis tick labels.
% The overall figure window size remains unchanged.
tl4.OuterPosition = [0.02 0.18 0.96 0.80];

% Left: equivalent estimands.
ax41 = nexttile(tl4);
hold(ax41,'on'); box(ax41,'on'); grid(ax41,'on');

xpos = 1:3;
dx = 0.07;

errorbar(ax41,xpos-dx,est_unc,1.96*se_unc,'o', ...
    'Color',cUn,'MarkerFaceColor',cUn, ...
    'MarkerSize',8,'LineWidth',1.8,'CapSize',10);
errorbar(ax41,xpos+dx,est_cent,1.96*se_cent,'s', ...
    'Color',cCe,'MarkerFaceColor',cCe, ...
    'MarkerSize',7,'LineWidth',1.8,'CapSize',10);

set(ax41,'XTick',1:3, ...
    'XTickLabel',{'Slope at \it{z}\rm = 0','Slope at mean(\it{z}\rm)','Interaction'}, ...
    'FontName','Arial','FontSize',22);
xtickangle(ax41,28);
ylabel(ax41,'Estimate (95% CI)');
title(ax41,'Estimates from both fits','FontWeight','normal');

lg41 = legend(ax41,{'Uncentered','Centered'}, ...
    'Location','northwest','Orientation','horizontal','Box','off');
lg41.FontName = 'Arial';
lg41.FontSize = 21;

% Sensible y limits with space for the legend.
allLow = [est_unc-1.96*se_unc; est_cent-1.96*se_cent];
allHigh = [est_unc+1.96*se_unc; est_cent+1.96*se_cent];
yr = max(allHigh)-min(allLow);
ylim(ax41,[min(allLow)-0.08*yr, max(allHigh)+0.18*yr]);
xlim(ax41,[0.55 3.45]);  % keep the first/last estimates away from the y-axes

% Right: VIFs.
ax42 = nexttile(tl4);
hold(ax42,'on'); box(ax42,'on'); grid(ax42,'on');

vif_mat = [vif1(:) vif2(:)];
hb4 = bar(ax42,vif_mat,'grouped','BarWidth',0.78,'LineWidth',0.5);
hb4(1).FaceColor = cUn;
hb4(2).FaceColor = cCe;
set(hb4,'EdgeColor','none');

set(ax42,'XTick',1:3, ...
    'XTickLabel',{'\it{x}','\it{z}','\it{x}\rm\cdot\it{z}'}, ...
    'FontName','Arial','FontSize',22);
ylabel(ax42,'VIF');
title(ax42,'VIFs','FontWeight','normal');

lg42 = legend(ax42,{'Uncentered','Centered'}, ...
    'Location','northwest','Orientation','horizontal','Box','off');
lg42.FontName = 'Arial';
lg42.FontSize = 21;

ylim(ax42,[0 1.18*max(vif_mat(:))]);

set(findall(fig4,'-property','FontName'),'FontName','Arial');

drawnow;
exportgraphics(fig4,fullfile(OUTPUT_DIR,'Figure4.png'), ...
    'Resolution',EXPORT_DPI,'BackgroundColor','white');

%% ========================================================================
%  SAVE REPLICATION RESULTS AND VERIFY FOUR FIGURES
%  ========================================================================
save(fullfile(OUTPUT_DIR,'replication_results.mat'), ...
    'Figure1_results','Figure2_results', ...
    'Figure3_MonteCarlo_results','Figure4_equivalent_estimands', ...
    'rmse_te','r2_te','k_best', ...
    'rmseAll','coefMSEAll','lassoNnz','N_MC','N_TEST', ...
    'b_ols','b_rdg','b_las','beta_true_std', ...
    'representative_replication','representative_seed', ...
    'representative_lasso_predictors','target_nnz','repScore', ...
    'vif1','vif2','kappa1','kappa2', ...
    'Figure3_counterchecks','MC','MC0');

nFigures = numel(findall(0,'Type','figure'));
fprintf('\nCreated %d figure windows.\n',nFigures);
if nFigures ~= 4
    warning('Expected exactly four figures, but found %d.',nFigures);
end

fprintf('PNG and numerical outputs saved to:\n  %s\n',OUTPUT_DIR);

%% ========================================================================
%  LOCAL HELPER FUNCTIONS
%  ========================================================================

function [Xz,yz,s] = standard_func(X,y,s)
    if nargin < 3
        s = struct('mX',mean(X),'sX',std(X), ...
                   'my',mean(y),'sy',std(y));
        s.sX(s.sX==0) = 1;
        if s.sy == 0
            s.sy = 1;
        end
    end
    Xz = (X-s.mX)./s.sX;
    yz = (y-s.my)/s.sy;
end

function k = ridgeCV_fast_func(X,y,grid,K)
    % Faster ridge CV:
    % each fold is standardized once, and all penalties are evaluated
    % together in a single call to ridge().
    cv = cvpartition(size(X,1),'KFold',K);
    mse = zeros(1,numel(grid));

    for i = 1:cv.NumTestSets
        tr = training(cv,i);
        te = test(cv,i);

        [Xt,yt,s] = standard_func(X(tr,:),y(tr));
        [Xv,yv] = standard_func(X(te,:),y(te),s);

        B = ridge(yt,Xt,grid,0);  % (p+1) x numel(grid)
        pred = Xv*B(2:end,:) + ones(size(Xv,1),1)*B(1,:);
        mse = mse + mean((yv-pred).^2,1);
    end

    mse = mse/cv.NumTestSets;
    [~,idx] = min(mse);
    k = grid(idx);
end

function MC = figure3_run_mc_func(seedOffset,N_MC,useParallel,n,p,Sigma, ...
        beta_true,eps_sd,N_TEST,ridge_grid,K_FOLDS,LASSO_NUM_LAMBDA,LASSO_LAMBDA_RATIO)
    % Runs N_MC replications of the Figure 3 design. Replication s uses
    % seed seedOffset + s, so results are identical in serial and parallel.
    rmse = nan(N_MC,3);
    coef_mse = nan(N_MC,3);
    coef_mse_raw = nan(N_MC,3);
    lasso_nnz = nan(N_MC,1);
    rmse_lasso_min = nan(N_MC,1);
    lasso_min_nnz = nan(N_MC,1);
    coef_mse_zero = nan(N_MC,1);

    if useParallel
        parfor s = 1:N_MC
            R = figure3_replication_func(seedOffset+s,n,p,Sigma,beta_true,eps_sd, ...
                N_TEST,ridge_grid,K_FOLDS,LASSO_NUM_LAMBDA,LASSO_LAMBDA_RATIO);
            rmse(s,:) = R.rmse;
            coef_mse(s,:) = R.coef_mse;
            coef_mse_raw(s,:) = R.coef_mse_raw;
            lasso_nnz(s) = R.lasso_nnz;
            rmse_lasso_min(s) = R.rmse_lasso_min;
            lasso_min_nnz(s) = R.lasso_min_nnz;
            coef_mse_zero(s) = R.coef_mse_zero;
        end
    else
        for s = 1:N_MC
            R = figure3_replication_func(seedOffset+s,n,p,Sigma,beta_true,eps_sd, ...
                N_TEST,ridge_grid,K_FOLDS,LASSO_NUM_LAMBDA,LASSO_LAMBDA_RATIO);
            rmse(s,:) = R.rmse;
            coef_mse(s,:) = R.coef_mse;
            coef_mse_raw(s,:) = R.coef_mse_raw;
            lasso_nnz(s) = R.lasso_nnz;
            rmse_lasso_min(s) = R.rmse_lasso_min;
            lasso_min_nnz(s) = R.lasso_min_nnz;
            coef_mse_zero(s) = R.coef_mse_zero;
        end
    end

    MC = struct('rmse',rmse,'coef_mse',coef_mse,'coef_mse_raw',coef_mse_raw, ...
        'lasso_nnz',lasso_nnz,'rmse_lasso_min',rmse_lasso_min, ...
        'lasso_min_nnz',lasso_min_nnz,'coef_mse_zero',coef_mse_zero);
end


function R = figure3_replication_func(seed,n,p,Sigma,beta_true,eps_sd,N_TEST, ...
        ridge_grid,K_FOLDS,LASSO_NUM_LAMBDA,LASSO_LAMBDA_RATIO)
    % One fully reproducible replication of the Figure 3 design. Used both
    % for the Monte Carlo analysis and for the representative replication.
    rng(seed,'twister');

    % Fresh training sample: the estimation problem remains n = 80.
    X_train = mvnrnd(zeros(1,p),Sigma,n);
    y_train = X_train*beta_true + eps_sd*randn(n,1);
    [Xtr_s,ytr_s,st_s] = standard_func(X_train,y_train);

    % OLS (the training data are centered, so the intercept is zero).
    bo = Xtr_s \ ytr_s;

    % Ridge, tuned only on the training sample (minimum CV error).
    ks = ridgeCV_fast_func(X_train,y_train,ridge_grid,K_FOLDS);
    Br = ridge(ytr_s,Xtr_s,ks,0);
    br = Br(2:end);
    b0r = Br(1);

    % Lasso, tuned only on the training sample using the 1-SE rule. The
    % solution at the minimum CV error, from the same fit, is kept as a
    % countercheck.
    [BLs,infos] = lasso(Xtr_s,ytr_s,'CV',K_FOLDS, ...
        'Standardize',false,'NumLambda',LASSO_NUM_LAMBDA, ...
        'LambdaRatio',LASSO_LAMBDA_RATIO);
    bl = BLs(:,infos.Index1SE);
    b0l = infos.Intercept(infos.Index1SE);
    blm = BLs(:,infos.IndexMinMSE);
    b0lm = infos.Intercept(infos.IndexMinMSE);

    % Large fresh independent test sample. It is standardized using the
    % TRAINING statistics so that predictions are evaluated on the same scale.
    X_test = mvnrnd(zeros(1,p),Sigma,N_TEST);
    y_test = X_test*beta_true + eps_sd*randn(N_TEST,1);
    [Xte_s,yte_s] = standard_func(X_test,y_test,st_s);

    yh = [Xte_s*bo, ...
          Xte_s*br+b0r, ...
          Xte_s*bl+b0l];

    SSE_te = sum((yte_s-yh).^2,1);
    R.rmse = sqrt(SSE_te/N_TEST)*st_s.sy;
    R.r2 = 1 - (SSE_te*st_s.sy^2) / sum((y_test-mean(y_test)).^2);
    R.rmse_lasso_min = sqrt(mean((yte_s-(Xte_s*blm+b0lm)).^2))*st_s.sy;

    % Coefficient MSE on the training-standardized coefficient scale.
    btrue_s = beta_true .* st_s.sX(:) / st_s.sy;
    R.coef_mse = [mean((bo-btrue_s).^2), ...
                  mean((br-btrue_s).^2), ...
                  mean((bl-btrue_s).^2)];
    R.coef_mse_zero = mean(btrue_s.^2);   % all-zero estimate (reference value)

    % Coefficient MSE on the original (unstandardized) scale.
    toRaw = st_s.sy ./ st_s.sX(:);
    R.coef_mse_raw = [mean((bo.*toRaw-beta_true).^2), ...
                      mean((br.*toRaw-beta_true).^2), ...
                      mean((bl.*toRaw-beta_true).^2)];

    R.lasso_nnz = nnz(abs(bl)>1e-6);
    R.lasso_min_nnz = nnz(abs(blm)>1e-6);
    R.lasso_idx = find(abs(bl)>1e-6);

    % Coefficients of this replication (used for the representative panel).
    R.bo = bo;
    R.br = br;
    R.bl = bl;
    R.b0r = b0r;
    R.b0l = b0l;
    R.ks = ks;
    R.btrue_s = btrue_s;
end


function k = bkw_condition_number_func(X)
    col_norms = sqrt(sum(X.^2));
    X_scaled = X./col_norms;
    s = svd(X_scaled);
    k = max(s)/min(s);
end

function [b,se,t,p,df,V] = ols_with_se_func(X,y)
    [n,k] = size(X);
    b = X\y;
    e = y-X*b;
    s2 = (e'*e)/(n-k);

    XtX = X'*X;
    XtX_inv = XtX\eye(k);
    V = s2*XtX_inv;

    se = sqrt(diag(V));
    t = b./se;
    df = n-k;
    p = 2*tcdf(-abs(t),df);
end

function v = vif_func(X)
    p = size(X,2);
    v = nan(p,1);

    for j = 1:p
        yj = X(:,j);
        Xj = [ones(size(X,1),1) X(:,setdiff(1:p,j))];
        bj = Xj\yj;
        rj = yj-Xj*bj;
        R2 = 1-(rj'*rj)/sum((yj-mean(yj)).^2);
        v(j) = 1/(1-R2);
    end
end
