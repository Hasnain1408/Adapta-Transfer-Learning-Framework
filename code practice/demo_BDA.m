function [Acc, acc_ite, A] = demo_BDA()
% Small built-in demo so users can run `BDA` without arguments.

	% Same input data source as TCA_demo.m
	X_src_table = readtable('Datasets/RELINK/A.xlsx');
	X_src_all = table2array(X_src_table);

	X_tar_table = readtable('Datasets/RELINK/S.xlsx');
	X_tar_all = table2array(X_tar_table);

	options.gamma = 1.0;
	options.lambda = 0.1;
	options.kernel_type = 'linear';
	options.T = 10;
	options.dim = min(100,size(X_src_all,2)-1);
	options.mu = 0;
	options.mode = 'W-BDA';

	% BDA accepts TCA-style domain matrices when last column is label.
	[Acc, acc_ite, A] = BDA(X_src_all, X_tar_all, options);
	fprintf('Acc: %.4f\n', Acc);
end