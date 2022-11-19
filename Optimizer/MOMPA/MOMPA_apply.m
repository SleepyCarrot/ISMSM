function [POS_fit,POS] = MOMPA_apply(func_name,var_min,var_max,nObj,Np,maxiter)
% func_name函数句柄
% var_min变量下限
% var_max变量上限
% nObj优化维数
% Np种群数量
% maxiter最大迭代数

if nargin == 3
    Np = 50;
    maxiter = 100;
end

dim = size(var_min,2); % 变量数
nInt = [1]; % 整数变量在设计变量中的位置，可设为空[]

% MOMPA algorithm
[POS_fit,POS]=IMOMPA2(Np,maxiter,var_min,var_max,dim,func_name,nObj,nInt);
end