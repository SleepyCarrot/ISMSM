% Multi-objective function

clear all; clc; 
% close all;
% MultiObjFnc = 'Schaffer';
% MultiObjFnc = 'Kursawe';
% MultiObjFnc = 'Poloni';
% MultiObjFnc = 'Viennet2';
% MultiObjFnc = 'Viennet3';
MultiObjFnc = 'ZDT1';
% MultiObjFnc = 'ZDT2';
% MultiObjFnc = 'ZDT3';
% MultiObjFnc = 'ZDT6';

switch MultiObjFnc
    case 'Schaffer'         % Schaffer
        fobj = @(x) [x(:).^2, (x(:)-2).^2];
        nobj = 2;
        dim = 1;
        lb = -5;
        ub = 5;
        load('Schaffer.mat');
    case 'Kursawe'          % Kursawe 
        fobj = @(x) [-10.*(exp(-0.2.*sqrt(x(:,1).^2+x(:,2).^2)) + exp(-0.2.*sqrt(x(:,2).^2+x(:,3).^2))), ...
                             sum(abs(x).^0.8 + 5.*sin(x.^3),2)];
        nobj = 2;
        dim = 3;
        lb = -5.*ones(1,dim);
        ub = 5.*ones(1,dim);
        load('Kursawe.mat');
    case 'Poloni'           % Poloni's two-objective
        A1 = 0.5*sin(1)-2*cos(1)+sin(2)-1.5*cos(2);
        A2 = 1.5*sin(1)-cos(1)+2*sin(2)-0.5*cos(2);
        B1 = @(x,y) 0.5.*sin(x)-2.*cos(x)+sin(y)-1.5.*cos(y);
        B2 = @(x,y) 1.5.*sin(x)-cos(x)+2.*sin(y)-0.5.*cos(y);
        f1 = @(x,y) 1+(A1-B1(x,y)).^2+(A2-B2(x,y)).^2;
        f2 = @(x,y) (x+3).^2+(y+1).^2;
        fobj = @(x) [f1(x(:,1),x(:,2)), f2(x(:,1),x(:,2))];
        nobj = 2;
        dim = 2;
        lb = -pi.*ones(1,dim);
        ub = pi.*ones(1,dim);
        
    case 'Viennet2'         % Viennet2
        f1 = @(x,y) 0.5.*(x-2).^2+(1/13).*(y+1).^2+3;
        f2 = @(x,y) (1/36).*(x+y-3).^2+(1/8).*(-x+y+2).^2-17;
        f3 = @(x,y) (1/175).*(x+2.*y-1).^2+(1/17).*(2.*y-x).^2-13;
        fobj = @(x) [f1(x(:,1),x(:,2)), f2(x(:,1),x(:,2)), f3(x(:,1),x(:,2))];
        nobj = 3;
        dim = 2;
        lb = [-4, -4];
        ub = [4, 4];
        load('Viennet2.mat');
    case 'Viennet3'         % Viennet3
        f1 = @(x,y) 0.5.*(x.^2+y.^2)+sin(x.^2+y.^2);
        f2 = @(x,y) (1/8).*(3.*x-2.*y+4).^2 + (1/27).*(x-y+1).^2 +15;
        f3 = @(x,y) (1./(x.^2+y.^2+1))-1.1.*exp(-(x.^2+y.^2));
        fobj = @(x) [f1(x(:,1),x(:,2)), f2(x(:,1),x(:,2)), f3(x(:,1),x(:,2))];
        nobj = 3;
        dim = 2;
        lb = [-3, -10];
        ub = [10, 3];
        load('Viennet3.mat');
    case 'ZDT1'             % ZDT1 (convex)
        g = @(x) 1+9.*sum(x(:,2:end),2)./(size(x,2)-1);
        fobj = @(x) [x(:,1), g(x).*(1-sqrt(x(:,1)./g(x)))];
        nobj = 2;
        dim = 30; 
        lb = zeros(1,dim);
        ub = ones(1,dim);
        load('ZDT1.mat');
    case 'ZDT2'             % ZDT2 (non-convex)
        f = @(x) x(:,1);
        g = @(x) 1+9.*sum(x(:,2:end),2)./(size(x,2)-1);
        h = @(x) 1-(f(x)./g(x)).^2;
        fobj = @(x) [f(x), g(x).*h(x)];
        nobj = 2;
        dim = 30; 
        lb = zeros(1,dim);
        ub = ones(1,dim);
        load('ZDT2.mat');
    case 'ZDT3'             % ZDT3 (discrete)
        f = @(x) x(:,1);
        g  = @(x) 1+(9/size(x,2)-1).*sum(x(:,2:end),2);
        h  = @(x) 1 - sqrt(f(x)./g(x)) - (f(x)./g(x)).*sin(10.*pi.*f(x));
        fobj = @(x) [f(x), g(x).*h(x)];
        nobj = 2;
        dim = 30;
        lb = 0.*ones(1,dim);
        ub = 1.*ones(1,dim);
        load('ZDT3.mat');
    case 'ZDT6'             % ZDT6 (non-uniform)
        f = @(x) 1 - exp(-4.*x(:,1)).*sin(6.*pi.*x(:,1));
        g = @(x) 1 + 9.*(sum(x(:,2:end),2)./(size(x,2)-1)).^0.25;
        h = @(x) 1 - (f(x)./g(x)).^2;
        fobj = @(x) [f(x), g(x).*h(x)];
        nobj = 2;
        dim = 10;
        lb = 0.*ones(1,dim);
        ub = 1.*ones(1,dim);
        load('ZDT6.mat');
end

% MOMPA algorithm 
SearchAgents_no=60;
Max_iteration=100;
nInt=[];
tic
[Best_score,Best_pos]=MOMPA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj,nobj,nInt);
toc
feature('memstats')

if exist('PF')
    if size(PF,2)==2
        plot(PF(:,1),PF(:,2),'.','color',0.8.*ones(1,3)); hold on;
    elseif size(PF,2)==3
        plot3(PF(:,1),PF(:,2),PF(:,3),'.','color',0.8.*ones(1,3)); hold on;
    end
end