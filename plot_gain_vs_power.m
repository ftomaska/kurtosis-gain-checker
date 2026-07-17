%PLOT_GAIN_VS_POWER  Reproduce the PTC-scatter and photon-flux-beeswarm
% figures across laser-power conditions, from kurtosis_checker.py's
% "Export Results" (gain_results_summary.csv + per-run detail CSVs).
%
% This is a plain script, not a function -- edit the CONFIG section
% below (folder path, condition labels/order/colors) and press Run.
%
% Expected files, all in one folder (from "one shared folder, distinct
% labels" -- i.e. you clicked Export Results once per power condition,
% into the same destination folder each time):
%
%   gain_results_summary.csv          one row per exported condition
%   <label>_<timestamp>_ptc_bins.csv        one row per PTC bin
%   <label>_<timestamp>_flux_percell.csv    one row per cell (if that
%                                            run had cell traces)
%
% gain_results_summary.csv columns (written by write_gain_export() in
% kurtosis_checker.py -- see GAIN_SUMMARY_FIELDS there if this ever
% drifts out of sync):
%   timestamp, label, source, n_cells, cnmf_used, fs_hz, enf,
%   fit_lo_pct, fit_hi_pct, spatial_bin_px, edge_margin_px, gain_true,
%   slope, intercept, r2, flux_med_photons_per_s, flux_sem_photons_per_s,
%   use_baseline_for_flux, baseline_pctile, ptc_bins_file, flux_percell_file
%
% Produces two figures:
%   Figure 1 -- "panel b" style: variance vs. mean raw fluorescence,
%               one PTC scatter block per condition side by side
%               (each block its own local 0..max(mu) range), with the
%               fitted shot-noise line (var = slope*mu + intercept)
%               overlaid per condition.
%   Figure 2 -- "panel f" style: photon flux per cell per second, one
%               beeswarm per condition, with the per-condition median
%               labeled next to a short marker line. No d' (GCaMP6s)
%               axis -- that conversion isn't something the app
%               computes; add your own and a `yyaxis right` block in
%               the Figure 2 section below once you have it.

clear; clc;

% ======================= CONFIG -- edit this ==========================
folder = uigetdir(pwd, 'Select the folder with gain_results_summary.csv');
if isequal(folder, 0)
    error('plot_gain_vs_power:noFolder', 'No folder selected.');
end

% conditions(i).label must match EXACTLY (case-sensitive) the label you
% typed into the app's "Export Results" dialog for that run.
conditions = struct( ...
    'label',   {'65mW', '110mW', '155mW', '180mW'}, ...
    'display', {'65 mW', '110 mW', '155 mW', '180 mW'});

colors = [ ...
    0.20 0.40 0.75;   % blue
    0.80 0.25 0.20;   % red/orange
    0.90 0.65 0.10;   % gold
    0.55 0.30 0.75];  % purple

% Set true to save each figure as a PNG next to the source CSVs.
save_figures = false;
% ========================================================================

n_cond = numel(conditions);
if size(colors, 1) < n_cond
    error('plot_gain_vs_power:notEnoughColors', ...
        'Only %d colors defined for %d conditions -- add more rows to `colors`.', ...
        size(colors, 1), n_cond);
end

summary_path = fullfile(folder, 'gain_results_summary.csv');
if ~isfile(summary_path)
    error('plot_gain_vs_power:missingSummary', ...
        'Could not find gain_results_summary.csv in:\n  %s', folder);
end
% Found it: some power labels are plain digits with no "mW" suffix
% ("180", "155" in Filip's own export folder) -- indistinguishable from
% a numeric column to readtable's automatic type detection, which was
% silently importing `label` as double (turning the alphanumeric rows
% like "65mW" into NaN) rather than as text. That's what regexp() was
% actually choking on, not a char/string/cell mismatch. Force `label`
% (and the other genuinely-text columns) to string explicitly via
% detectImportOptions so this never depends on what the values happen
% to look like.
opts = detectImportOptions(summary_path, 'TextType', 'string');
text_cols = {'timestamp', 'label', 'source', 'ptc_bins_file', 'flux_percell_file'};
for tc = 1:numel(text_cols)
    if any(strcmp(opts.VariableNames, text_cols{tc}))
        opts = setvartype(opts, text_cols{tc}, 'string');
    end
end
summary = readtable(summary_path, opts);

% Look up each condition's row up front (fail fast with a clear message
% before any plotting starts if a label is missing/misspelled).
%
% Matched by the leading numeric power value, not the raw label string
% verbatim -- real exports vary in case and whether "mW" is even present
% (e.g. "65mW", "110mw", "180", "155" all seen from the same export
% folder), so an exact strcmp() against 'conditions(i).label' is too
% brittle and was rejecting valid rows.
cond_power = nan(n_cond, 1);
for i = 1:n_cond
    tok = regexp(conditions(i).label, '\d+', 'match', 'once');
    if ~isempty(tok)
        cond_power(i) = str2double(tok);
    end
end

row_power = nan(height(summary), 1);
for r = 1:height(summary)
    tok = regexp(summary.label(r), '\d+', 'match', 'once');
    if strlength(tok) > 0
        row_power(r) = str2double(tok);
    end
end

rows = cell(n_cond, 1);
for i = 1:n_cond
    hit = (row_power == cond_power(i));
    match = summary(hit, :);
    if isempty(match)
        present_str = strjoin(summary.label, ', ');
        error('plot_gain_vs_power:labelNotFound', ...
            ['No row in gain_results_summary.csv has a power matching "%s".\n' ...
             'Labels present: %s'], conditions(i).label, present_str);
    end
    if height(match) > 1
        warning('plot_gain_vs_power:duplicateLabel', ...
            ['%d rows share label "%s" (exported more than once) -- ' ...
             'using the most recent (last) one.'], height(match), ...
            conditions(i).label);
        match = match(end, :);
    end
    rows{i} = match;
end

% Print the fitted gain for each condition (gain_true, from the PTC
% slope in gain_results_summary.csv) to the command window.
fprintf('\nFitted gain by condition:\n');
for i = 1:n_cond
    fprintf('  %-10s gain = %8.2f  (slope = %.2f, R^2 = %.4f)\n', ...
        conditions(i).display, rows{i}.gain_true, rows{i}.slope, rows{i}.r2);
end
fprintf('\n');

% ============ Figure 1: PTC variance vs. mean raw fluorescence ========
fig1 = figure('Name', 'PTC: variance vs. mean raw fluorescence', ...
              'Color', 'w');
ax1 = axes(fig1); hold(ax1, 'on');

gap_frac    = 0.15;   % extra gap left between blocks, as a fraction of block width
x_offset    = 0;
tick_pos    = [];
tick_lbl    = {};
label_x     = zeros(n_cond, 1);   % where to put each condition's name (2nd pass)

for i = 1:n_cond
    row = rows{i};
    bins_path = fullfile(folder, row.ptc_bins_file(1));
    if ~isfile(bins_path)
        warning('plot_gain_vs_power:missingBins', ...
            'Bins file for "%s" not found, skipping that block:\n  %s', ...
            conditions(i).label, bins_path);
        continue
    end
    bins = readtable(bins_path);

    mu = bins.mu_bin;
    va = bins.var_bin;
    in_fit = strcmpi(strtrim(string(bins.in_fit)), 'true');

    mu_max  = max(mu);
    block_w = mu_max * (1 + gap_frac);
    x = mu + x_offset;

    % excluded bins dimmer, included bins solid, same color per condition
    scatter(ax1, x(~in_fit), va(~in_fit), 14, colors(i, :), 'filled', ...
        'MarkerFaceAlpha', 0.25, 'MarkerEdgeColor', 'none');
    scatter(ax1, x(in_fit), va(in_fit), 14, colors(i, :), 'filled', ...
        'MarkerFaceAlpha', 0.55, 'MarkerEdgeColor', 'none');

    % fitted shot-noise line: var = slope * mu + intercept
    fit_x = linspace(0, mu_max, 2);
    fit_y = row.slope * fit_x + row.intercept;
    plot(ax1, fit_x + x_offset, fit_y, '-', 'Color', colors(i, :), ...
        'LineWidth', 1.4);

    tick_pos(end+1) = x_offset;                    %#ok<AGROW>
    tick_lbl{end+1} = '0';                          %#ok<AGROW>
    tick_pos(end+1) = x_offset + mu_max;            %#ok<AGROW>
    tick_lbl{end+1} = sprintf('%.0f', mu_max);      %#ok<AGROW>

    if i < n_cond
        xline(ax1, x_offset + block_w, '--', 'Color', [0.6 0.6 0.6]);
    end

    label_x(i) = x_offset + 0.03 * mu_max;
    x_offset = x_offset + block_w;
end

% second pass: condition-name labels at a consistent height, now that
% the axes have seen all the data (avoids each label being placed at a
% y that later data pushes past)
yl = ylim(ax1);
label_y = yl(1) + 0.97 * (yl(2) - yl(1));
for i = 1:n_cond
    if label_x(i) == 0 && i > 1
        continue  % skipped block (missing file)
    end
    lbl_text = sprintf('%s\ngain = %.1f', conditions(i).display, rows{i}.gain_true);
    text(ax1, label_x(i), label_y, lbl_text, ...
        'Color', colors(i, :), 'FontWeight', 'bold', ...
        'VerticalAlignment', 'top');
end

xlabel(ax1, 'Mean raw fluorescence (each condition offset to its own block)');
ylabel(ax1, 'Variance');
ax1.XTick = tick_pos;
ax1.XTickLabel = tick_lbl;
box(ax1, 'off');
title(ax1, 'Photon transfer curve by laser power');

if save_figures
    exportgraphics(fig1, fullfile(folder, 'ptc_by_power.png'), 'Resolution', 200);
end

% ================= Figure 2: photon-flux beeswarm ======================
fig2 = figure('Name', 'Photon flux per cell per second', 'Color', 'w');
ax2 = axes(fig2); hold(ax2, 'on');

has_swarmchart = ~isempty(which('swarmchart'));
if ~has_swarmchart
    warning('plot_gain_vs_power:noSwarmchart', ...
        ['swarmchart() not found (needs MATLAB R2020b+) -- falling back ' ...
         'to plain jittered scatter, which does not guarantee ' ...
         'non-overlapping points the way a true beeswarm does.']);
end

for i = 1:n_cond
    row = rows{i};
    flux_file = row.flux_percell_file(1);
    if isempty(char(flux_file))
        % char() folds both a truly empty string ("") and readtable's
        % <missing> placeholder (for a blank CSV cell) down to '', so
        % this one check catches "no flux file for this condition"
        % regardless of which one readtable produced.
        warning('plot_gain_vs_power:noFlux', ...
            'Condition "%s" has no per-cell flux file (no cell traces for that run) -- skipped.', ...
            conditions(i).label);
        continue
    end
    flux_path = fullfile(folder, flux_file);
    if ~isfile(flux_path)
        warning('plot_gain_vs_power:missingFlux', ...
            'Flux file for "%s" not found, skipping:\n  %s', ...
            conditions(i).label, flux_path);
        continue
    end
    flux = readtable(flux_path);
    y = flux.photon_flux_per_s;

    % log axis can't render non-positive values -- drop them here
    % rather than let MATLAB silently omit them from the plot only.
    n_nonpos = sum(y <= 0);
    if n_nonpos > 0
        warning('plot_gain_vs_power:nonPositiveFlux', ...
            '%d of %d cells in "%s" have photon_flux_per_s <= 0 and are omitted from the log-scale plot.', ...
            n_nonpos, numel(y), conditions(i).label);
        y = y(y > 0);
    end

    if has_swarmchart
        swarmchart(ax2, repmat(i, size(y)), y, 18, colors(i, :), ...
            'filled', 'MarkerFaceAlpha', 0.6, 'XJitterWidth', 0.6);
    else
        jitter = (rand(size(y)) - 0.5) * 0.6;
        scatter(ax2, i + jitter, y, 18, colors(i, :), 'filled', ...
            'MarkerFaceAlpha', 0.6);
    end

    med = median(y, 'omitnan');
    plot(ax2, [i - 0.3, i + 0.3], [med, med], 'k-', 'LineWidth', 1.5);
    text(ax2, i + 0.35, med, sprintf('%.0f', med), 'FontSize', 9, ...
        'VerticalAlignment', 'middle');
end

ax2.XTick = 1:n_cond;
ax2.XTickLabel = {conditions.display};
xlabel(ax2, 'Laser power');
ylabel(ax2, 'Photons/cell/s');
xlim(ax2, [0.5, n_cond + 0.5]);
ax2.YScale = 'log';
box(ax2, 'off');
title(ax2, 'Photon flux by laser power');

if save_figures
    exportgraphics(fig2, fullfile(folder, 'flux_by_power.png'), 'Resolution', 200);
end
