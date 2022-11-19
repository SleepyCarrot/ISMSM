function [PEMF_Error, model] = improved_PEMF(trainer,X,Y,error_type,alfa,n_steps)
% based on version 2016.v1
%  Predictive Estimation of Model Fidelity (PEMF) is a model-independent 
%  approach to quantify surrogate model error.  PEMF takes as input a
%  model trainer, sample data on which to train the model, and hyper-
%  parameter values to apply to the model.  As output, it provides an 
%  estimate of the median or maximum error in the surrogate model.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 北京理工大学改写
% 输入参数：
%        X - 样本集x，每行包含一个样本点的输入数据
%        Y - 样本集y，每行包含一个样本点的输出数据，目前只考虑单输出模型
%        trainer - 由X和Y训练得到的代理模型的训练函数句柄
%        error_type - 'mod_med','mod_max','var_med','var_max',
%                     'pred_med','pred_max'
%        alfa - 误差预测模型的虚拟样本集点数扩增比例（由N个点变为N+αN个）
%        n_steps - 误差预测模型的预测步数
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    For further explaination of PEMF's optional inputs, refer to the 
%       PEMF paper. "Predictive quantification of surrogate model fidelity
%       based on modal variations with sample density." by Chowdhury and 
%       Mehmani. DOI: 10.1007/s00158-015-1234-z
%    The above article and its citation (bibtex) can be found at:
%       http://adams.eng.buffalo.edu/publications/
%    
%    Cite PEMF as:
%       A. Mehmani, S. Chowdhury, and A. Messac, "Predictive quantification
%       of surrogate model fidelity based on modal variations with sample 
%       density," Structural and Multidisciplinary Optimization, vol. 52, 
%       pp. 353-373, 2015.
%    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n_pnts = size(X,1);  % 训练点数
n_var = size(X,2);  % 设计变量个数

% 输入参数检查
switch nargin
    case 3
        error_type = ["mod_med","mod_max"];
        alfa = 0.1;
        n_steps = 4;
    case 4
        alfa = 0.1;
        n_steps = 4;
    case 5
        n_steps = 4;
end

% 预测模型设值
n_pnts_final = floor(max(alfa*n_pnts,3));  % 最后一次迭代加入的训练点数
n_pnts_step = n_pnts_final;  % 误差预测模型的步长
n_permutations = 40;  % 每一步尝试的组合数

% 参数格式检查
model = check_input(trainer, X,Y,error_type,n_pnts_step,n_steps,n_pnts);

%% PEMF
disp('PEMF Starting');

% 由X得到设计变量上下限
LB = zeros(n_var,1);
UB = zeros(n_var,1);
for j=1:n_var
    LB(j)=min(X(:,j));
    UB(j)=max(X(:,j));
end

data=[X,Y];

PEMF_Error_max_mod = zeros(1,n_steps);
PEMF_Error_med_mod = zeros(1,n_steps);
PEMF_Error_max_var = zeros(1,n_steps);
PEMF_Error_med_var = zeros(1,n_steps);
n_train = zeros(1,n_steps); % 训练点数
MedianTest = zeros(n_steps, n_permutations);
MaxTest = zeros(n_steps, n_permutations);
med_params = zeros(n_steps,2);
max_params = zeros(n_steps,2);


for i=1:n_steps
    n_train(i) = n_pnts-(n_pnts_final+(i-1)*n_pnts_step);
    % 第i步中所有组合的训练和测试点
    M_Combination = zeros(n_permutations,n_train(i));
    for i_NC = 1:n_permutations
        M_Combination(i_NC,:)=randsample(n_pnts,n_train(i));
    end
    
    % 中间代理的训练和测试
    for j=1:n_permutations
        % 分离训练集和测试集
        training_data = data(M_Combination(j,:),:);
        test_data = data;
        test_data(M_Combination(j,:),:)=[]; % 删去训练集数据行
        
        % 分离训练点和测试点的X、Y
        x_train = training_data(:,1:n_var);
        y_train = training_data(:,n_var+1);
        n_tests = size(test_data,1);
        x_test = test_data(:,1:n_var);
        y_test = test_data(:,n_var+1);
        
        % 模型训练和测试
        trained_model = trainer(x_train,y_train);
        
        RAE = zeros(1,n_tests); % RAE - Relative Absolute Error,相对绝对误差
        for k = 1:n_tests
            y_predicted = trained_model(x_test(k,:));           
            RAE(k) = abs((y_test(k)-y_predicted)/y_test(k));
        end
        
        % 计算RAE的中值和最大值
        MedianTest(i,j)=   median(RAE);
        MaxTest(i,j)   =   max(RAE); 
    
    end
    
    % MODE-MED,VAR-MED
    if ismember('mod_med',error_type) || ismember('var_med',error_type) || ismember('pred_med',error_type)
        % 删除Med（RAE）中的异常值并拟合到对数正态
        parmhat = lognfit_outliers(MedianTest(i,:),70); 
        med_params(i,:) = parmhat;
        % 计算模态值
        PEMF_Error_med_mod(i)=exp(parmhat(1)-(parmhat(2))^2);
        % 计算标准差
        PEMF_Error_med_var(i)=sqrt((exp((parmhat(2))^2)-1)*exp(2*parmhat(1)+(parmhat(2))^2));
    end
    % MOD-Max,VAR-MAX
    if ismember('mod_max',error_type) || ismember('var_max',error_type) || ismember('pred_max',error_type)
        % 删除Max(RAE)中的异常值并拟合到对数正态
        parmhat = lognfit_outliers(MaxTest(i,:),60); 
        max_params(i,:) = parmhat;
        % Mode of Max Estimation
        PEMF_Error_max_mod(i)=exp(parmhat(1)-(parmhat(2))^2);
        % 计算标准差
        PEMF_Error_max_var(i)=sqrt((exp((parmhat(2))^2)-1)*exp(2*parmhat(1)+(parmhat(2))^2));
    end

    tot = n_steps*n_permutations;
    curr = i*n_permutations;
%     fprintf('Iter %d: %d of %d intermediate models evaluated\n', ...
%         i, curr, tot);
end

n_train=flipud(n_train(:));
PEMF_Error_med_mod=flipud(PEMF_Error_med_mod(:));
PEMF_Error_max_mod=flipud(PEMF_Error_max_mod(:));
PEMF_Error_med_var=flipud(PEMF_Error_med_var(:));
PEMF_Error_max_var=flipud(PEMF_Error_max_var(:));
MaxTest = flipud(MaxTest);
MedianTest = flipud(MedianTest);
max_params = flipud(max_params);
med_params = flipud(med_params);

%% 选择误差的最佳拟合模型并预测总误差
for model_type = 1:2
    [RMe,~,~]=SelectRegression(n_train,PEMF_Error_med_mod,model_type,n_pnts);
    RMSE_MedianE(model_type,:)=RMe(:);
    [RMa,~,~]=SelectRegression(n_train,PEMF_Error_max_mod,model_type,n_pnts);
    RMSE_MaxE(model_type,:)=RMa(:);
    [RMev,~,~]=SelectRegression(n_train,PEMF_Error_med_var,model_type,n_pnts);
    RMSE_MedianEv(model_type,:)=RMev(:);
    [RMav,~,~]=SelectRegression(n_train,PEMF_Error_max_var,model_type,n_pnts);
    RMSE_MaxEv(model_type,:)=RMav(:);
end

%%  误差中值模态值
[~,model_id] = min(RMSE_MedianE);
model_type_med = model_id;
[~,MedianPrediction,v_med] = SelectRegression(n_train,PEMF_Error_med_mod,model_type_med,n_pnts,alfa);
CorrelationParameterMedian = SmoothnessCriteria(n_train,PEMF_Error_med_mod,model_type_med);
if abs(CorrelationParameterMedian)>=0.90
    PEMF_MedError_return = MedianPrediction(1);
    PEMF_MedError_pred_return = MedianPrediction(2);
    x_med=[n_train;n_pnts];
else
    PEMF_MedError_return = PEMF_Error_med_mod(n_steps);
    PEMF_MedError_pred_return = PEMF_MedError_return;
    x_med=[n_train;n_train(end)];
    if ismember('mod_med',error_type)
        fprintf('\nSmoothness criterion violated for modal predicition of median error.\n')
        fprintf('K-fold estimate is used from last iteration.\n\n');
    end
end

%%  误差最大值模态值
[~,model_id] = min(RMSE_MaxE);
model_type_max = model_id;
[~,MaxPrediction,v_max] = SelectRegression(n_train,PEMF_Error_max_mod,model_type_max,n_pnts,alfa);
CorrelationParameterMax = SmoothnessCriteria(n_train,PEMF_Error_max_mod,model_type_max);
if abs(CorrelationParameterMax)>=0.90
    PEMF_MaxError_return = MaxPrediction(1);
    PEMF_MaxError_pred_return = MedianPrediction(2);
    x_max=[n_train;n_pnts];
else
    PEMF_MaxError_return = PEMF_Error_max_mod(n_steps);
    PEMF_MaxError_pred_return = PEMF_MaxError_return;
    x_max=[n_train;n_train(end)];
    if ismember('mod_max',error_type)
        fprintf('\nSmoothness criterion violated for modal predicition of maximum error.\n')
        fprintf('K-fold estimate is used from last iteration.\n\n');
    end
end

%%  误差中值标准差
[~,model_id] = min(RMSE_MedianEv);
model_type_medv = model_id;
[~,MedianPredictionv,v_medv] = SelectRegression(n_train,PEMF_Error_med_var,model_type_medv,n_pnts);
CorrelationParameterMedianv = SmoothnessCriteria(n_train,PEMF_Error_med_var,model_type_medv);
if abs(CorrelationParameterMedianv)>=0.90
    PEMF_MedErrorv_return = MedianPredictionv;
    x_medv=[n_train;n_pnts];
else
    PEMF_MedErrorv_return = PEMF_Error_med_var(n_steps);
    x_medv=[n_train;n_train(end)];
    if ismember('var_med',error_type)
        fprintf('\nSmoothness criterion violated for standard deviation predicition of median error.\n')
        fprintf('K-fold estimate is used from last iteration.\n\n');
    end
end

%%  误差最大值标准差
[~,model_id] = min(RMSE_MaxEv);
model_type_maxv = model_id;
[~,MaxPredictionv,v_maxv] = SelectRegression(n_train,PEMF_Error_max_var,model_type_maxv,n_pnts);
CorrelationParameterMaxv = SmoothnessCriteria(n_train,PEMF_Error_max_var,model_type_maxv);
if abs(CorrelationParameterMaxv)>=0.90
    PEMF_MaxErrorv_return = MaxPredictionv;
    x_maxv=[n_train;n_pnts];
else
    PEMF_MaxErrorv_return = PEMF_Error_max_var(n_steps);
    x_maxv=[n_train;n_train(end)];
    if ismember('var_max',error_type)
        fprintf('\nSmoothness criterion violated for standard deviation predicition of maximum error.\n')
        fprintf('K-fold estimate is used from last iteration.\n\n');
    end
end

% %% 绘图
% PEMF_Error_med_mod=[PEMF_Error_med_mod;PEMF_MedError_return];
% PEMF_Error_max_mod=[PEMF_Error_max_mod;PEMF_MaxError_return];
% PEMF_Error_med_var=[PEMF_Error_med_var;PEMF_MedErrorv_return];
% PEMF_Error_max_var=[PEMF_Error_max_var;PEMF_MaxErrorv_return];
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% if(ismember('mod_med',error_type))
%     fig = plot_pemf(x_med,PEMF_Error_med_mod,med_params,model_type_med,v_med);
%     set(get(get(fig,'CurrentAxes'),'Title'),'String','误差中值(Modal)')
% end
% if(ismember('mod_max',error_type))
%     fig = plot_pemf(x_max,PEMF_Error_max_mod,max_params,model_type_max,v_max);
%     set(get(get(fig,'CurrentAxes'),'Title'),'String','误差最大值(Modal)')
% end
% if(ismember('var_med',error_type))
%     fig = plot_pemf(x_medv,PEMF_Error_med_var,med_params,model_type_medv,v_medv);
%     set(get(get(fig,'CurrentAxes'),'Title'),'String','误差中值(Standard Deviation)')
% end
% if(ismember('var_max',error_type))
%     fig = plot_pemf(x_maxv,PEMF_Error_max_var,max_params,model_type_maxv,v_maxv);
%     set(get(get(fig,'CurrentAxes'),'Title'),'String','误差最大值(Standard Deviation)')
% end
% %%%%%%%%%%%%%%%%%%%%%%%%%%
%% 返回所需要的误差
PEMF_Error = [];
if(ismember('mod_med',error_type))
    PEMF_Error(end+1) = PEMF_MedError_return;
    fprintf('\nPEMF_Error (median_modal): %f\n\n',PEMF_MedError_return)
end
if(ismember('mod_max',error_type))
    PEMF_Error(end+1) = PEMF_MaxError_return;
    fprintf('\nPEMF_Error (max_modal): %f\n\n',PEMF_MaxError_return)
end
if(ismember('var_med',error_type))
    PEMF_Error(end+1) = PEMF_MedErrorv_return;
    fprintf('\nPEMF_Error (median_standard_deviation): %f\n\n',PEMF_MedErrorv_return)
end
if(ismember('var_max',error_type))
    PEMF_Error(end+1) = PEMF_MaxErrorv_return;
    fprintf('\nPEMF_Error (max_standard_deviation): %f\n\n',PEMF_MaxErrorv_return)
end
if(find(error_type=='pred_med'))
    PEMF_Error(end+1) = PEMF_MedError_pred_return;
    fprintf('\nPEMF_Error (median_modal_predict): %f\n\n',PEMF_MedError_pred_return)
end
if(find(error_type=='pred_max'))
    PEMF_Error(end+1) = PEMF_MaxError_pred_return;
    fprintf('\nPEMF_Error (max_predict): %f\n\n',PEMF_MaxError_pred_return)
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 参数格式检查
function [model] = check_input(trainer,X,Y,error_type,n_pnts_step,n_steps,n_pnts)
% 检查X和Y是否长度匹配
if(n_pnts ~= size(Y,1))
    error('请保证X数组和Y数组列数相同。\n');
end

% 检查样本数量是否足够
if(n_pnts < n_steps*n_pnts_step + 3)
    error('样本数量过少，请添加样本数，减小虚拟样本集扩增比例或减少步数。\n');
end

% % 检查trainer是否为函数句柄
if(~isa(trainer,'function_handle'))
    error('trainer需为函数句柄。');
end

% 测试trainer是否能被调用，以及是否返回正确格式
try
    model = trainer(X,Y);
catch ME
    msg = 'trainer格式不匹配，无法正确被调用。\n';
    causeException = MException('MATLAB:myCode:dimensions',msg);
    ME = addCause(ME,causeException);
    rethrow(ME)
end

% 检查trainer输出是否符合y = model(x)
try
    model(X(1,:));
catch ME
    msg = 'trainer输出格式不匹配，无法正确被调用。\n';
    causeException = MException('MATLAB:myCode:dimensions',msg);
    ME = addCause(ME,causeException);
    rethrow(ME)
end

% 检查误差类型列表长度
if length(error_type) < 2
    warning('所选择的误差类型过少。\n');
end

% 检查误差类型是否为'mod_med','mod_max','var_med'或'var_max'
for error_n = error_type
    if(~(strcmp(error_n,'mod_med') || strcmp(error_n,'mod_max') || ...
        strcmp(error_n,'var_med') || strcmp(error_n,'var_max') || ...
        strcmp(error_n,'pred_med') || strcmp(error_n,'pred_max')))
        error('误差类型需要为''mod_med'',''mod_max'',''var_med'',''var_max'',''pred_med''或''pred_max''。\n');
    end
end

% 检查Y是否为单列数据
if(size(Y,2) ~= 1)
    error('Y应该为单列数据。\n');
end

% 当数据跨度过大时警告
if( min(Y)/max(Y) < 5*10^-3)
    warning('数据跨度远远超过2个数量级，可能对相对误差度量造成困难。\n');
end

end

% 删除异常值并拟合到对数正态
function parmhat = lognfit_outliers(data,outlier_percent)
    % 删除百分位数以上的异常值
    YP = prctile(data,outlier_percent);  
    i=1;
    CT = zeros(length(data));
    for k=1:max(length(data))
        if data(k) <= YP
            CT(i)= data(k);
            i=i+1;
        end
    end
    CT(i:end) = [];

    % 计算平均值、标准差
    CP = CT';
    mu = mean (CP);
    sigma = std (CP);
    [n,~] = size(CP);
    Meanmat = repmat(mu,n,1);
    Sigmamat = repmat(sigma,n,1);
    % 删除3σ置信区间外的点
    outliers = abs(CP-Meanmat) > 3*Sigmamat;
    CP(any(outliers,2),:) = [];
    % 删除零点
    CP(find(CP==0))=[];
    % 对数正态拟合，返回均值和标准差
    parmhat = lognfit(CP(:)); 
end

% 选择回归模型
function [RMSE,ErrorPrediction,VCoe] = SelectRegression(X,Y,iSType,NinP,alfa)
% Helper function with a few modes
% - Able to fit PEMF error to two different regression models depending on
%       the value of isType 
% - Able to predict the next value in the regression (ErrorPrediction)
% - Able to give the model fit parameters (VCoe = [a,b])
X(isnan(Y(:,1)),:)=[];
Y(isnan(Y(:,1)),:)=[];

%% 1. 指数回归函数 Exponential Fit Model   Y=a*exp(b*X)
if iSType==1
    ff = fit(X,Y,'exp1');
    a11=ff.a;
    b11=ff.b;
    switch nargin
        case 4
            ErrorPrediction=a11*exp(b11*NinP);
        case 5
            ErrorPrediction=[a11*exp(b11*NinP), a11*exp(b11*floor(NinP*(1+alfa)))];
    end
    
    data=Y;
    for j=1:max(size(X))
       estimate(j)=a11*exp(b11*X(j)); 
    end
    
    % 计算RMSE
    RMSE=Grmse(data,estimate');
    VCoe=[a11,b11];
    
end
%% 2. 乘法回归函数 Power Fit Model   Y=a*X^b

if iSType==2
     [a11,b11]=PowerFit(Y,X);
     switch nargin
     	case 4
            ErrorPrediction=a11*(NinP)^(b11);
     	case 5
            ErrorPrediction=[a11*(NinP)^(b11), a11*(floor(NinP*(1+alfa)))^(b11)];
     end
     
    data=Y;
    for j=1:max(size(X))
       estimate(j)=a11*X(j)^(b11); 
    end
     
	% 计算RMSE
    RMSE=Grmse(data,estimate');
    VCoe=[a11,b11];
    
end
end 

% 乘法回归Power Fit Model
function [a,b] = PowerFit(Y,X)
% Performs a regression fit with a Power fit model Y = a*X^b

n=length(X);
Z=zeros(1,n);
for i=1:n
    Z(i)=log(Y(i));
end
w=zeros(1,n);
for i=1:n
    w(i)=log(X(i));
end
wav=sum(w)/n;
zav=sum(Z)/n;
sum(Z);
Swz=0;
Sww=0;
for i=1:n
    Swz=Swz +w(i)*Z(i)-wav*zav;
    Sww=Sww + (w(i))^2-wav^2;
end

a1=Swz/Sww;
a0=zav-a1*wav;
a=exp(a0);
b=a1;

xp=(0:0.001:max(X));
yp=zeros(1,length(xp));
for i=1:length(xp)
    yp(i)=a.*(xp(i)^b);
end

end 

% 计算RMSE
function r = Grmse(data,estimate)
% Function to calculate root mean square error from a data vector or matrix 
% I = ~isnan(data) & ~isnan(estimate); 
% data = data(I); estimate = estimate(I);
rI=0;
for I=1:max(size(data))
    rI=rI+(data(I)-estimate(I)).^2;
end
RI=rI/(max(size(data)));
r=sqrt(RI);

end 

% 平滑度标准（相关系数）
function Rho = SmoothnessCriteria(x,y,iSType)
% Returns the smoothness of the fit for a given regression model
if iSType==1
    [R,~] = corrcoef(x,log(y));
end

if iSType==2
    [R,~] = corrcoef(log(x),log(y));
end

Rho = R(1,2);
end

% 绘图
function fig = plot_pemf(X,Y, lognfit_params, reg_type, reg_fit_param)

fig = figure;
hold on;

mu1 = lognfit_params(1,1);
sig1 = lognfit_params(1,2);
mode_x1 = exp(mu1-sig1^2);

% create, rotate, and scale distributions
npnts = 100; step = X(2)-X(1);
for i=1:1:length(lognfit_params)
    mu = lognfit_params(i,1);
    sig = lognfit_params(i,2);
    mode_x = exp(mu-sig^2);
    mode_p = lognpdf(mode_x,mu,sig);
    plot_tune = 0.05;
    xmax = fzero(@(x)lognpdf(x,mu,sig)-plot_tune*mode_p,[mode_x,1000*mode_x]);
    xmax = min(xmax,3*mode_x1);
    xs = 0:xmax/(npnts-1):xmax;
    for j = 1:1:length(xs)
        ys(j) = lognpdf(xs(j),mu,sig);
    end
    step_frac = 0.75;
    ys = ys.*(step_frac*step/max(ys));
    pnts = [xs',ys'];
    R = [cosd(90),sind(90);-sind(90),cosd(90)]; % rotates 90 deg
    pnts = pnts*R; % rotates 90 deg
    pnts(:,1) = pnts(:,1)+X(i);
    gray = 249/255*[1 1 1];
    patch([pnts(1,1),pnts(:,1)',pnts(1,1)],[pnts(2,2),pnts(:,2)',pnts(end,2)],gray,'EdgeColor','w','LineStyle',':');
    dist = plot(pnts(:,1),pnts(:,2),'k:');
    plot([pnts(1,1),pnts(1,1)],[pnts(2,2),pnts(end,2)],'k-.');
    
end

xs = X(1):0.01:X(end);
ys = 0*xs;
a = reg_fit_param(1); b = reg_fit_param(2);
if(reg_type == 1)
    ys = a.*exp(b.*xs);
elseif(reg_type == 2)
    ys = a.*(xs).^(b);
end

fitted = plot(xs,ys,'--','LineWidth',2);

calc = plot(X(1:end-1),Y(1:end-1),'bo','LineWidth',2); 
pred = plot(X(end),Y(end),'g+','LineWidth',2,'MarkerSize',8);  

hold off;
legend([dist,calc,pred],'误差分布','中间代理误差估值','最终代理误差估值');
title('PEMF Error'); ylabel('相对误差（绝对值）');
if(Y(end) == Y(end-1))
    n_points = X(end) + step;
else
    n_points = X(end);
end
s = strcat({'训练点数(总训练点数为'},num2str(n_points),{')'});
xlabel(s{1});
     
end