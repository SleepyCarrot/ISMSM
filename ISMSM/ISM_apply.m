function [model_name, models, added_points, points_value] = ISM_apply()
clear all; clc;
Fnc = 'Branin-Hoo';   %其他
% Fnc = 'Hartmann';     %其他
% Fnc = 'Dixon  _Price';  %谷形
% Fnc = 'N_Perm';       %其他
% Fnc = 'Ackley';          %许多局部最小值
% Fnc = 'Schaffer2';    %许多局部最小值
% Fnc = 'Bohachevsky1'; %碗形
% Fnc = 'Booth';        %盘形
% Fnc = 'Rosenbro  ck';   %谷形
% Fnc = 'De Jong5';     %陡坡形
% Fnc = 'Easom';        %陡坡形
% Fnc = 'Powell';       %其他
Test = 'Test1';
% Test = 'Test2';
% Test = 'Test3';
 
datanum=30; 
np=6;
maxgen=7;

switch Fnc 
    case 'Branin-Hoo'
        fun = @(x)Brainin_Hoo(x);
        nVar = 2;
        var_min = [-5,0];
        var_max = [10,15];
        
    case 'Hartmann'
        fun = @(x)Hartmann(x);
        nVar = 6;
        var_min = 0;
        var_max = 1;
          
    case  'Dixon_Price'
        fun = @(x)Dixon_Price(x);
        nVar = 30;
        var_min = -10;
        var_max = 10;
        
    case  'N_Perm'
        fun = @(x)N_Perm(x);
        nVar = 20;
        var_min = -20;
        var_max = 21;
        
    case  'Ackley'
        fun = @(x)Ackley(x);
        nVar = 4;
        var_min = -32.768;
        var_max = 32.768;
        
    case  'Schaffer2'
        fun = @(x)Schaffer2(x);
        nVar = 2;
        var_min = -100;
        var_max = 100;
        
    case  'Bohachevsky1'
        fun = @(x)Boha1(x);
        nVar = 2;
        var_min = -100;
        var_max = 100;
        
    case  'Booth'
        fun = @(x)Booth(x);
        nVar = 2;
        var_min = -10;
        var_max = 10;

    case  'Rosenbrock'
        fun = @(x)Rosen(x);
        nVar = 3;
        var_min = -5;
        var_max = 10;
        
    case 'De Jong5'
        fun = @(x)DJ5(x);
        nVar = 2;
        var_min = -65.536;
        var_max = 65.536;
        
    case 'Easom'
        fun = @(x)Easom(x);
        nVar = 2;
        var_min = -100;
        var_max = 100;
        
    case 'Powell'
        fun = @(x)Powell(x);
        nVar = 8;
        var_min = -4;
        var_max = 5;

end

X = lhsdesign(datanum,nVar);
X = X.*(var_max-var_min)+var_min;
Y=fun(X);


switch Test
%     case 'Test1'
%         [model_name, models, added_points,truePF0,truePF1,truePF2] = ISM(X,Y,["mod_med","mod_max"],0.05,4,...
%         40,50,0.9,0.5,0.05,10,20,0.4,2,2,20,5,0.5);
%     case 'Test2'
%         [model_name, models, added_points,truePF0,truePF1,truePF2] = ISM(X,Y,["mod_med","var_med"],0.05,4,...
%         40,50,0.9,0.5,0.05,10,20,0.4,2,2,20,5,0.5);
%     case 'Test3'
%         [model_name, models, added_points,truePF0,truePF1,truePF2] = ISM(X,Y,["mod_med","pred_med"],0.2,4,...
%         40,50,0.9,0.5,0.05,10,20,0.4,2,2,20,5,0.5);
    case 'Test1'
        [model_name, models, added_points, points_value] = ISM(X,Y,["mod_med","mod_max"],0.05,4,...
        np,maxgen,10,20,0.4,2,2,20,5,0.5);
    case 'Test2'
        [model_name, models, added_points, points_value] = ISM(X,Y,["mod_med","var_med"],0.05,4,...
        np,maxgen,10,20,0.4,2,2,20,5,0.5);
    case 'Test3'
        [model_name, models, added_points, points_value] = ISM(X,Y,["mod_med","pred_med"],0.2,4,...
        np,maxgen,10,20,0.4,2,2,20,5,0.5);
end
save('result.mat','model_name','models', 'added_points');
end

function [y] = Brainin_Hoo(x)
for i = 1:length(x)
    y(i,:) = branin(x(i,:));
end
end

function [y] = Hartmann(x)
c = [1.0,1.2,3.0,3.2]';
A = [10,3,17,3.5,1.7,8;
     0.05,10,17,0.1,8,14;
     3,3.5,1.7,10,17,8;
     17,8,0.05,10,0.1,14];
P = 10^(-4)*[1312,1696,5569,124,8283,5886;
             2329,4135,8307,3736,1004,9991;
             2348,1451,3522,2883,3047,6650;
             4047,8828,8732,5743,1091,381];
         
for k = 1:length(x)
    xx = x(k,:);
    outer = 0;
    for i = 1:4
        inner = 0;
        for j = 1:6
            xj = xx(j);
            Aij = A(i,j);
            Pij = P(i,j);
            inner = inner+Aij*(xj-Pij)^2;
        end
        new = c(i)*exp(-inner);
        outer = outer+new;
    end
    y(k,:) = -(2.58+outer)/1.94;
end
end

function [y] = Dixon_Price(x)
for i = 1:length(x)
    xx = x(i,:);
    x1 = xx(1);
    d = length(xx);
    term1 = (x1-1)^2;
    sum = 0;
    for j = 2:d
        xi = xx(j);
        xold = xx(j-1);
        new = j*(2*xi^2-xold)^2;
        sum = sum+new;
    end
    y(i,:) = term1+sum;
end
end

function [y] = N_Perm(x)
for i = 1:length(x)
    xx = x(i,:);
    d = 20;
    outer = 0;
    for k = 1:d
        inner = 0;
        for j = 1:d
            xj = xx(j);
            inner = inner+(j^k+0.5)*((xj/j)^k-1);
        end
        outer = outer+inner^2;
    end
    y(i,:) = outer;
end
end

function [y] = Ackley(x)
a = 20;
b = 0.2;
c = 2*pi;
d = 4;
for i = 1:length(x)
	xx = x(i,:);
	sum1 = 0;
	sum2 = 0;
	for ii = 1:4
		xi = xx(ii);
		sum1 = sum1 + xi^2;
		sum2 = sum2 + cos(c*xi);
	end
	term1 = -a * exp(-b*sqrt(sum1/d));
	term2 = -exp(sum2/d);
	y(i,:) = term1 + term2 + a + exp(1);
end
end

function [y] = Schaffer2(x)
for i = 1:length(x)
	xx = x(i,:);
	x1 = xx(1);
	x2 = xx(2);
	fact1 = (sin(x1^2-x2^2))^2 - 0.5;
	fact2 = (1 + 0.001*(x1^2+x2^2))^2;
	y(i,:) = 0.5 + fact1/fact2;
end
end

function [y] = Boha1(x)
for i = 1:length(x)
	xx = x(i,:);
	x1 = xx(1);
	x2 = xx(2);
	term1 = x1^2;
	term2 = 2*x2^2;
	term3 = -0.3 * cos(3*pi*x1);
	term4 = -0.4 * cos(4*pi*x2);
	y(i,:) = term1 + term2 + term3 + term4 + 0.7;
end
end

function [y] = Booth(x)
for i = 1:length(x)
	xx = x(i,:);
	x1 = xx(1);
	x2 = xx(2);
	term1 = (x1 + 2*x2 - 7)^2;
	term2 = (2*x1 + x2 - 5)^2;
	y(i,:) = term1 + term2;
end
end

function [y] = Rosen(x)
for i = 1:length(x)
	xx = x(i,:);
	sum = 0;
	for ii = 1:(3-1)
		xi = xx(ii);
		xnext = xx(ii+1);
		new = 100*(xnext-xi^2)^2 + (xi-1)^2;
		sum = sum + new;
	end
	y(i,:) = sum;
end
end

function [y] = DJ5(x)
for i = 1:length(x)
	xx = x(i,:);
	x1 = xx(1);
	x2 = xx(2);
	sum = 0;
	A = zeros(2, 25);
	a = [-32, -16, 0, 16, 32];
	A(1, :) = repmat(a, 1, 5);
	ar = repmat(a, 5, 1);
	ar = ar(:)';
	A(2, :) = ar;

	for ii = 1:25
		a1i = A(1, ii);
		a2i = A(2, ii);
		term1 = ii;
		term2 = (x1 - a1i)^6;
		term3 = (x2 - a2i)^6;
		new = 1 / (term1+term2+term3);
		sum = sum + new;
	end
	y(i,:) = 1 / (0.002 + sum);
end
end

function [y] = Easom(x)
for i = 1:length(x)
	xx = x(i,:);
	x1 = xx(1);
	x2 = xx(2);
	fact1 = -cos(x1)*cos(x2);
	fact2 = exp(-(x1-pi)^2-(x2-pi)^2);
	y(i,:) = fact1*fact2;
end
end

function [y] = Powell(x)
for i = 1:length(x)
	xx = x(i,:);
	sum = 0;
	for ii = 1:(8/4)
		term1 = (xx(4*ii-3) + 10*xx(4*ii-2))^2;
		term2 = 5 * (xx(4*ii-1) - xx(4*ii))^2;
		term3 = (xx(4*ii-2) - 2*xx(4*ii-1))^4;
		term4 = 10 * (xx(4*ii-3) - xx(4*ii))^4;
		sum = sum + term1 + term2 + term3 + term4;
	end
	y(i,:) = sum;
end
end