function [X_src_new, X_tar_new, A] = TCA_demo()
    % Small built-in demo so users can run `TCA` without arguments.
    % Load source data from A.xlsx
    X_src_table = readtable('Datasets/RELINK/A.xlsx');
    X_src = table2array(X_src_table);  % Convert table to matrix

    % Load target data from S.xlsx
    X_tar_table = readtable('Datasets/RELINK/S.xlsx');
    X_tar = table2array(X_tar_table);  % Convert table to matrix

    % Display sizes to verify
%     fprintf('Source data size: %d samples x %d features\n', size(X_src));
%     fprintf('Target data size: %d samples x %d features\n', size(X_tar));

    % Set TCA options
    options.lambda = 1.0;
    options.dim = min(30, size(X_src, 2));  % Ensure dim <= n_features
    options.kernel_type = 'linear';  % or 'primal', 'rbf'
    options.gamma = 1.0;  % only needed for 'rbf'

    % Run TCA
    [X_src_new, X_tar_new, A] = TCA(X_src, X_tar, options);

    % Check results
%     fprintf('Transformed source size: %d x %d\n', size(X_src_new));
%     fprintf('Transformed target size: %d x %d\n', size(X_tar_new));
end