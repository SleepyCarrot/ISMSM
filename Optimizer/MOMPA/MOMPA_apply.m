function [POS_fit,POS] = MOMPA_apply(func_name,var_min,var_max,nObj,Np,maxiter)
% func_name�������
% var_min��������
% var_max��������
% nObj�Ż�ά��
% Np��Ⱥ����
% maxiter��������

if nargin == 3
    Np = 50;
    maxiter = 100;
end

dim = size(var_min,2); % ������
nInt = [1]; % ������������Ʊ����е�λ�ã�����Ϊ��[]

% MOMPA algorithm
[POS_fit,POS]=IMOMPA2(Np,maxiter,var_min,var_max,dim,func_name,nObj,nInt);
end