function [POS_fit,POS] = MOPSO_apply(func_name,var_min,var_max,Np,maxgen,W,C1,C2,ngrid,maxvel,u_mut)
% Parameters
% params.Np = 200;        % Population size
% params.Nr = 200;        % Repository size
% params.maxgen = 100;    % Maximum number of generations
% params.W = 0.4;         % Inertia weight
% params.C1 = 2;          % Individual confidence factor
% params.C2 = 2;          % Swarm confidence factor
% params.ngrid = 20;      % Number of grids in each dimension
% params.maxvel = 5;      % Maxmium vel in percentage
% params.u_mut = 0.5;     % Uniform mutation percentage

% f1 = @(x,y) 0.5.*(x.^2+y.^2)+sin(x.^2+y.^2);
% f2 = @(x,y) (1/8).*(3.*x-2.*y+4).^2 + (1/27).*(x-y+1).^2 +15;
% f3 = @(x,y) (1./(x.^2+y.^2+1))-1.1.*exp(-(x.^2+y.^2));
% MultiObj.fun = @(x) [f1(x(:,1),x(:,2)), f2(x(:,1),x(:,2)), f3(x(:,1),x(:,2))];
% MultiObj.nVar = 2;
% MultiObj.var_min = [-3, -10];
% MultiObj.var_max = [10, 3];
% MultiObj.IntVar = [1];

if nargin == 3
    Np = 200;
    maxgen = 100;
    W = 0.4;
    C1 = 2;
    C2 = 2;
    ngrid = 20;
    maxvel = 5;
    u_mut = 0.5;
end

params.Np = Np;
params.Nr = Np;
params.maxgen = maxgen;
params.W = W;
params.C1 = C1;
params.C2 = C2;
params.ngrid = ngrid;
params.maxvel = maxvel;
params.u_mut = u_mut;
MultiObj.fun = func_name;
MultiObj.var_min = var_min;
MultiObj.var_max = var_max;
MultiObj.nVar = size(var_min,2);
MultiObj.IntVar = [1];

REP = MOPSO(params,MultiObj);
POS_fit = REP.pos_fit;
POS = REP.pos;
end