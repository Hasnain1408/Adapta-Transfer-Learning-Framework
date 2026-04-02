function [Acc, acc_ite, A] = demo_BDA()
% Small built-in demo so users can run `BDA` without arguments.

	% input data source 
	X_src_table = readtable('Datasets/NASA/CM1.xlsx');
	X_src_all = table2array(X_src_table);

	X_tar_table = readtable('Datasets/NASA/PC1.xlsx');
	X_tar_all = table2array(X_tar_table);

	options.gamma = 1.0;
	options.lambda = 0.1; %regularigation
	options.kernel_type = 'linear'; %primal, linear, rbf
	options.T = 1; % number of iterations
	options.dim = min(100,size(X_src_all,2)-1); % reduced dimensions
	options.mu = .6; % balance factor
	options.mode = 'W-BDA';


	[Acc, acc_ite, A] = BDA(X_src_all, X_tar_all, options);
    fprintf('Acc: %.4f\n', Acc);
%      fprintf('acc_ite: %.4f\n',acc_ite);
%      fprintf('A: %.4f\n', A);
    
    
end