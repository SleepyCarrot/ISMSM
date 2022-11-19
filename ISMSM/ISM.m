%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 北京理工大学
% 输入参数：
%        X - 样本集x，每行包含一个样本点的输入数据
%        Y - 样本集y，每行包含一个样本点的输出数据，目前只考虑单输出模型
%        error_type - 'mod_med','mod_max','var_med','var_max',
%                     'pred_med','pred_max'
%        alfa - 误差预测模型的虚拟样本集点数扩增比例（由N个点变为N+αN个）
%        n_steps - 误差预测模型的预测步数
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [model_name, models, added_points, points_value, errors]=ISM(X,Y,...
error_type,alfa,n_steps,Np1,maxgen1,Np2,maxgen2,W,C1,C2,ngrid,maxvel,u_mut)

%% 添加所有子路径到工作路径
here = mfilename('fullpath');
[path, ~, ~] = fileparts(here);
addpath(genpath(path));

%% 判断输入参数是否符合要求
if(nargin < 4)
    error('输入参数过少，请检查是否已输入待拟合数据X、Y、误差类型和迭代步数。');
end

%% 用RSM排除不合适的拟合阶数
HP0 = [1,7];
HP1 = [1,21];
HP2 = [2,3];

% %测试函数
% x0 = lhsdesign(30,2);
% for i = 1:length(x0)
%     x(i,:) = [-5,0] + x0(i,:)*15;
%     y(i,:) = branin(x(i,:));
%     y(i,:) = sum(x(i,:).^2+3+2*x(i,1)+x(i,2).^3);
% end
% X = x;
% Y = y;
% error_type = ["mod_med","mod_max"];
% alfa = 0.05;
% n_steps = 4;
% Np1=5;maxgen1=10;pc=0.9;pm=0.5;ms=0.05;
% Np2=5;maxgen2=10;W=0.4;C1=2;C2=2;ngrid=20;maxvel=5;u_mut=0.5;

% 由X得到设计变量上下限
n_var = size(X,2);
n_pnts = size(X,1);
nObj = size(error_type,2);
LB = zeros(n_var,1);
UB = zeros(n_var,1);
for j = 1:n_var
    LB(j) = min(X(:,j));
    UB(j) = max(X(:,j));
end

% 随机生成20%测试点和80%训练点
train_pnts =randsample(n_pnts,round(0.8*n_pnts));
train_X = X(train_pnts,:);
train_Y = Y(train_pnts,:);
test_X = X;
test_X(train_pnts,:)=[];
test_Y = Y;
test_Y(train_pnts,:)=[];
mean_Y = mean(test_Y);
SS_tot = sum((test_Y-mean_Y).^2);

% 建立1~4阶RSM代理模型并计算R^2
% 计算RSM-4的R2值
surrogate_trainer = rsm_trainer(4,train_X,train_Y);
surr_Y = surrogate_trainer(test_X);
SS_res = sum((surr_Y-test_Y).^2);
R2 = 1-SS_res/SS_tot;
if R2 < 0.9
    HP0(1) = 5;
else
    % 计算RSM-3的R2值
    surrogate_trainer = rsm_trainer(3,train_X,train_Y);
    surr_Y = surrogate_trainer(test_X);
    SS_res = sum((surr_Y-test_Y).^2);
    R2 = 1-SS_res/SS_tot;
    if R2 < 0.9
        HP0(1) = 4;
    else
        % 计算RSM-2的R2值
        surrogate_trainer = rsm_trainer(2,train_X,train_Y);
        surr_Y = surrogate_trainer(test_X);
        SS_res = sum((surr_Y-test_Y).^2);
        R2 = 1-SS_res/SS_tot;
        if R2 < 0.9
            HP0(1) = 3;
        else
            % 计算RSM-1的R2值
            surrogate_trainer = rsm_trainer(1,train_X,train_Y);
            surr_Y = surrogate_trainer(test_X);
            SS_res = sum((surr_Y-test_Y).^2);
            R2 = 1-SS_res/SS_tot;
            if R2 < 0.9
                HP0(1) = 2;
            end
        end
    end
end

% 建立0~2阶Kriging代理模型并计算R^2
% 计算Kriging-2-Lin的R2值
try
    surrogate_trainer = krig_trainer(15,train_X,train_Y,1);
    surr_Y = surrogate_trainer(test_X);
    SS_res = sum((surr_Y-test_Y).^2);
    R2 = 1-SS_res/SS_tot;
catch
    R2 = 0;
end
if R2 < 0.9
    HP1(2) = 15;
else
    % 计算Kriging-1-Lin的R2值
    surrogate_trainer = krig_trainer(9,train_X,train_Y,1);
    surr_Y = surrogate_trainer(test_X);
    SS_res = sum((surr_Y-test_Y).^2);
    R2 = 1-SS_res/SS_tot;
    if R2 < 0.9
        HP1(1) = 13;
    else
        % 计算Kriging-0-Lin的R2值
        surrogate_trainer = krig_trainer(3,train_X,train_Y,1);
        surr_Y = surrogate_trainer(test_X);
        SS_res = sum((surr_Y-test_Y).^2);
        R2 = 1-SS_res/SS_tot;
        if R2 < 0.9
            HP1(1) = 7;
        end
    end
end

%% 在筛选后的模型池内进行优化
n_error_type = size(error_type,2);
LB0 = [HP0(1)];
UB0 = [HP0(2)];
LB1 = [HP1(1),0];
UB1 = [HP1(2),1];
LB2 = [HP2(1),0,0];
UB2 = [HP2(2),1,1];

trainer0 = @(var)HP0_trainer(var,X,Y,error_type,alfa,n_steps);
trainer1 = @(var)HP1_trainer(var,X,Y,error_type,alfa,n_steps);
trainer2 = @(var)HP2_trainer(var,X,Y,error_type,alfa,n_steps);

if n_error_type < 3
%     [POS_fit0,POS0,truePF0] = NSGAII_apply(trainer0,LB0,UB0,nObj,Np1,maxgen1,...
%     pc,pm,ms);
%     [POS_fit1,POS1,truePF1] = NSGAII_apply(trainer1,LB1,UB1,nObj,Np1,maxgen1,...
%     pc,pm,ms);
%     [POS_fit2,POS2,truePF2] = NSGAII_apply(trainer2,LB2,UB2,nObj,Np1,maxgen1,...
%     pc,pm,ms);
    [POS_fit0,POS0] = MOMPA_apply(trainer0,LB0,UB0,nObj,Np1,maxgen1);
    [POS_fit1,POS1] = MOMPA_apply(trainer1,LB1,UB1,nObj,Np1,maxgen1);
    [POS_fit2,POS2] = MOMPA_apply(trainer2,LB2,UB2,nObj,Np1,maxgen1);
else
    [POS_fit0,POS0] = MOPSO_apply(trainer0,LB0,UB0,Np2,maxgen2,W,C1,C2,...
    ngrid,maxvel,u_mut);
    [POS_fit1,POS1] = MOPSO_apply(trainer1,LB1,UB1,Np2,maxgen2,W,C1,C2,...
    ngrid,maxvel,u_mut);
    [POS_fit2,POS2] = MOPSO_apply(trainer2,LB2,UB2,Np2,maxgen2,W,C1,C2,...
    ngrid,maxvel,u_mut);
end

%% 得到三个模型池的总Pareto前沿并绘图
[Pareto,Pareto_id] = ParetoFilter(-[POS_fit0;POS_fit1;POS_fit2]);
Pareto = -Pareto;

figure();
if(nObj == 2)
%     scatter(POS_fit0(:,1),POS_fit0(:,2),30,'filled','markerFaceAlpha',...
%     0.5,'MarkerFaceColor',[0 0.4470 0.7410]); 
%     hold on;
%     scatter(POS_fit1(:,1),POS_fit1(:,2),30,'filled','markerFaceAlpha',...
%     0.5,'MarkerFaceColor',[0.8500 0.3250 0.0980]); 
%     hold on;
%     scatter(POS_fit2(:,1),POS_fit2(:,2),30,'filled','markerFaceAlpha',...
%     0.5,'MarkerFaceColor',[0.9290 0.6940 0.1250]); 
%     hold on;

%     scatter(truePF0(:,1),truePF0(:,2),50,'ob','LineWidth',1.5); 
%     hold on;
%     scatter(truePF1(:,1),truePF1(:,2),50,'xg','LineWidth',1.5); 
%     hold on;
%     scatter(truePF2(:,1),truePF2(:,2),50,'+k','LineWidth',1.5); 
%     hold on;
    scatter(POS_fit0(:,1),POS_fit0(:,2),50,'ob','LineWidth',1.5); 
    hold on;
    scatter(POS_fit1(:,1),POS_fit1(:,2),50,'xg','LineWidth',1.5); 
    hold on;
    scatter(POS_fit2(:,1),POS_fit2(:,2),50,'+k','LineWidth',1.5); 
    hold on;
    
%     scatter(Pareto(:,1),Pareto(:,2),60,'ok'); 
    scatter(Pareto(:,1),Pareto(:,2),80,'sr','LineWidth',1.5); 
    hold on;
    xlabel(error_type(1)); 
    ylabel(error_type(2));
    legend('HP-0','HP-1','HP-2','Pareto');
    print(gcf,'-dpng','result.jpg','-r900');
    
elseif(nObj >= 3)
    plot3(POS_fit0(:,1),POS_fit0(:,2),POS_fit0(:,3),'.','Color',...
    [0 0.4470 0.7410],'MarkerSize',7); 
    hold on;
    plot3(POS_fit1(:,1),POS_fit1(:,2),POS_fit1(:,3),'.','Color',...
    [0.8500 0.3250 0.0980],'MarkerSize',7); 
    hold on;
    plot3(POS_fit2(:,1),POS_fit2(:,2),POS_fit2(:,3),'.','Color',...
    [0.9290 0.6940 0.1250],'MarkerSize',7); 
    hold on;
    plot3(Pareto(:,1),Pareto(:,2),Pareto(:,3),'ok','MarkerSize',7); 
    hold on;
    xlabel(error_type(1)); 
    ylabel(error_type(2));
    legend('HP-0','HP-1','HP-2','Pareto');
    print(gcf,'-dpng','result.jpg','-r900');
    
end
grid on

%% 得到解集对应的模型
n_Pareto = size(Pareto_id,1);
n_pos0 = size(POS0,1);
n_pos1 = size(POS1,1);
model0_set = [];model0_fit_set = [];
model1_set = [];model1_fit_set = [];
model2_set = [];model2_fit_set = [];

for i=1:n_Pareto
    if Pareto_id(i) <= n_pos0
        model0_set = [model0_set;POS0(Pareto_id(i))];
        model0_fit_set = [model0_fit_set;POS_fit0(Pareto_id(i),:)];
    elseif Pareto_id(i) <= n_pos0+n_pos1
        model1_set = [model1_set;POS1(Pareto_id(i)-n_pos0,:)];
        model1_fit_set = [model1_fit_set;POS_fit1(Pareto_id(i)-n_pos0,:)];
    else
        model2_set = [model2_set;POS2(Pareto_id(i)-n_pos0-n_pos1,:)];
        model2_fit_set = [model2_fit_set;POS_fit2(Pareto_id(i)-n_pos0-n_pos1,:)];
    end
end

model0_set = unique(model0_set,'rows');model0_fit_set = unique(model0_fit_set,'rows');
model1_set = unique(model1_set,'rows');model1_fit_set = unique(model1_fit_set,'rows');
model2_set = unique(model2_set,'rows');model2_fit_set = unique(model2_fit_set,'rows');

[model_name0,model0] = surr_trainer0(model0_set,X,Y,model0_fit_set);
[model_name1,model1] = surr_trainer1(model1_set,X,Y,model1_fit_set);
[model_name2,model2] = surr_trainer2(model2_set,X,Y,model2_fit_set);
model_name = [model_name0,model_name1,model_name2];
models = [model0,model1,model2];
model0_fit_set
model1_fit_set
model2_fit_set
errors = [model0_fit_set;model1_fit_set;model2_fit_set];

model0_set
model1_set
model2_set

%% 对优化模型集进行初步加点推荐
n_models = length(models);
added_points = [];
points_value = [];

for i = 1:n_models
    [model_value,model_pos] = MPA_apply(models{i},LB',UB',25,100,i);
    added_points = [added_points; model_pos];
    points_value = [points_value; model_value];
    display([num2str(i),' ',char(model_name(i)),':']);
    display(['    模型最终误差为: ',num2str(errors(i,:))]);
    display(['    建议加点位置: ', num2str(model_pos)]);
    display(['    该位置代理模型值为: ', num2str(model_value)]);
end

end


%% 子函数
% Pareto过滤器
function [p,id] = ParetoFilter(p)
[i, j] = size(p);
id = [1 : i]';
while i >= 1
    old_size = size(p,1);
    indices = sum(bsxfun(@ge,p(i,:),p),2) == j;
    indices(i) = false;
    p(indices,:) = [];
    id(indices) = [];
    i = i - 1 - (old_size - size(p,1)) + sum(indices(i:end));
end
end

% HP-0模型训练
function [model_name,model] = surr_trainer0(n,x,y,fit)
model = {};
model_name = [];
len = size(n,1);

for i = 1:len
    switch n(i)
        case {1,2,3,4}
            model_tmp = rsm_trainer(n(i),x,y);
            model_name_tmp = string(['RSM-',num2str(n(i))]);
                    
        case {5,6,7}
            model_tmp = rbf_trainer(n(i),x,y);
            
            switch n(i)
                case 5
                    model_name_tmp = "RBF-Linear";
                case 6
                    model_name_tmp = "RBF-Cubic";
                case 7
                    model_name_tmp = "RBF-Tps";
            end
    end
    model{end+1} = model_tmp;
%     display(['推荐模型类型为：',char(model_name_tmp)]);
    model_name = [model_name,model_name_tmp];
end
end

% HP-1模型训练
function [model_name,model] = surr_trainer1(var,x,y,fit)
model = {};
model_name = [];
len = size(var,1);

for i = 1:len
    n = var(i,1);
    var1 = var(i,2:end);
    if n <= 12 || n >= 16
        var1 = var1*(20-0.1)+0.1;
        model_tmp = krig_trainer(n,x,y,var1);
        
        switch n
            case {1,7,16}
                corr = 'EXP';
            case {2,8,17}
                corr = 'GAUSS';
            case {3,9,18}
                corr = 'LIN';
            case {4,10,19}
                corr = 'SPEHERICAL';
            case {5,11,20}
                corr = 'CUBIC'
            case {6,12,21}
                corr = 'SPLINE'
        end
        
        if n <= 6
            reg = '0';
        elseif n <=12
            reg = '1';
        else
            reg = '2';
        end
        
        model_name_tmp = string(['Kriging-',reg,'-',corr]);
            
    elseif n <= 14
        var1 = var1*(3-0.1)+0.1;
        model_tmp = rbf_trainer(n,x,y,var1);
        
        switch n
            case 13
                model_name_tmp = "RBF-Gaussian";
            case 14
                model_name_tmp = "RBF-Multiquadric";
        end
        
    else
        var1 = var1*(100-0.1)+0.1;
        model_tmp = svr_trainer(n,x,y,var1);
        
        model_name_tmp = "SVR-Linear";
        
    end
    
%     display(['推荐模型类型为：',char(model_name_tmp)]);
    model_name = [model_name,model_name_tmp];
    model{end+1} = model_tmp;
end
end

% HP-2模型训练
function [model_name,model] = surr_trainer2(var,x,y,fit)
model = {};
model_name = [];
len = size(var,1);

for i = 1:len
    n = var(i,1);
    var2 = var(i,2:end);
    var2(1) = var2(1)*(100-0.1)+0.1;
    var2(2) = var2(2)*(10-0.1)+0.1;
    model_tmp = svr_trainer(n,x,y,var2);
    model{end+1} = model_tmp;
    
    switch n
        case 1
            model_name_tmp = "SVR-Polynomial";
        case 2
            model_name_tmp = "SVR-RBF";
        case 3
            model_name_tmp = "SVR-Sigmoid";
    end
    
%     display(['推荐模型类型为：',char(model_name_tmp)]);
    model_name = [model_name,model_name_tmp];
end
end

% MPA寻最小值
function [Best_score,Best_pos]=MPA_apply(model,lb,ub,n_agents,max_iter,ii)
dim=length(lb);
[Best_score,Best_pos,Convergence_curve]=MPA(n_agents,max_iter,lb,ub,dim,...
model);

% 代理模型拓扑图
figure('Position',[500 400 700 290])
subplot(1,2,1);
x = [0:0.02:1]'*(ub-lb)+lb;
if dim == 1
    f = model(x);
    plot(x,f);
else
    for i=1:51
        for j=1:51
            f(i,j) = model([x(i,1),x(j,2),zeros(1,dim-2)]);
        end
    end
    surfc(x(:,1),x(:,2),f,'LineStyle','none');
    colormap winter;
end
title('Function Topology')
xlabel('x_1');
ylabel('x_2');
zlabel(['Surrogate Model','( x_1 , x_2 )'])

% 收敛曲线
subplot(1,2,2);
semilogy(Convergence_curve,'Color','r')
title('Objective space')
xlabel('Iteration');
ylabel('Best score obtained so far');
print(gcf,'-dpng',['model',num2str(ii),'.png']);
end