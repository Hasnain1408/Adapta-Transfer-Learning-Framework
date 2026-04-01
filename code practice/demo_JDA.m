function [Acc, acc_ite, A] = demo_JDA()
% Small built-in demo so users can run `JDA` without arguments.

    % Same input data source as TCA_demo.m
    X_src_table = readtable('Datasets/RELINK/A.xlsx');
    X_src_all = table2array(X_src_table);

    X_tar_table = readtable('Datasets/RELINK/S.xlsx');
    X_tar_all = table2array(X_tar_table);

    options.lambda = 1.0;
    options.dim = min(30, size(X_src_all, 2) - 1);
    options.kernel_type = 'linear';
    options.gamma = 1.0;
    options.T = 10;

    % JDA accepts TCA-style domain matrices when last column is label.
    [Acc, acc_ite, A] = JDA(X_src_all, X_tar_all, options);
    fprintf('Acc: %.4f\n', Acc);
end
