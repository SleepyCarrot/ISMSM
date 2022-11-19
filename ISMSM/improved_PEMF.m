function [PEMF_Error, model] = improved_PEMF(trainer,X,Y,error_type,alfa,n_steps)
% based on version 2016.v1
%  Predictive Estimation of Model Fidelity (PEMF) is a model-independent 
%  approach to quantify surrogate model error.  PEMF takes as input a
%  model trainer, sample data on which to train the model, and hyper-
%  parameter values to apply to the model.  As output, it provides an 
%  estimate of the median or maximum error in the surrogate model.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ��������ѧ��д
% ���������
%        X - ������x��ÿ�а���һ�����������������
%        Y - ������y��ÿ�а���һ���������������ݣ�Ŀǰֻ���ǵ����ģ��
%        trainer - ��X��Yѵ���õ��Ĵ���ģ�͵�ѵ���������
%        error_type - 'mod_med','mod_max','var_med','var_max',
%                     'pred_med','pred_max'
%        alfa - ���Ԥ��ģ�͵���������������������������N�����ΪN+��N����
%        n_steps - ���Ԥ��ģ�͵�Ԥ�ⲽ��
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

n_pnts = size(X,1);  % ѵ������
n_var = size(X,2);  % ��Ʊ�������

% ����������
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

% Ԥ��ģ����ֵ
n_pnts_final = floor(max(alfa*n_pnts,3));  % ���һ�ε��������ѵ������
n_pnts_step = n_pnts_final;  % ���Ԥ��ģ�͵Ĳ���
n_permutations = 40;  % ÿһ�����Ե������

% ������ʽ���
model = check_input(trainer, X,Y,error_type,n_pnts_step,n_steps,n_pnts);

%% PEMF
disp('PEMF Starting');

% ��X�õ���Ʊ���������
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
n_train = zeros(1,n_steps); % ѵ������
MedianTest = zeros(n_steps, n_permutations);
MaxTest = zeros(n_steps, n_permutations);
med_params = zeros(n_steps,2);
max_params = zeros(n_steps,2);


for i=1:n_steps
    n_train(i) = n_pnts-(n_pnts_final+(i-1)*n_pnts_step);
    % ��i����������ϵ�ѵ���Ͳ��Ե�
    M_Combination = zeros(n_permutations,n_train(i));
    for i_NC = 1:n_permutations
        M_Combination(i_NC,:)=randsample(n_pnts,n_train(i));
    end
    
    % �м�����ѵ���Ͳ���
    for j=1:n_permutations
        % ����ѵ�����Ͳ��Լ�
        training_data = data(M_Combination(j,:),:);
        test_data = data;
        test_data(M_Combination(j,:),:)=[]; % ɾȥѵ����������
        
        % ����ѵ����Ͳ��Ե��X��Y
        x_train = training_data(:,1:n_var);
        y_train = training_data(:,n_var+1);
        n_tests = size(test_data,1);
        x_test = test_data(:,1:n_var);
        y_test = test_data(:,n_var+1);
        
        % ģ��ѵ���Ͳ���
        trained_model = trainer(x_train,y_train);
        
        RAE = zeros(1,n_tests); % RAE - Relative Absolute Error,��Ծ������
        for k = 1:n_tests
            y_predicted = trained_model(x_test(k,:));           
            RAE(k) = abs((y_test(k)-y_predicted)/y_test(k));
        end
        
        % ����RAE����ֵ�����ֵ
        MedianTest(i,j)=   median(RAE);
        MaxTest(i,j)   =   max(RAE); 
    
    end
    
    % MODE-MED,VAR-MED
    if ismember('mod_med',error_type) || ismember('var_med',error_type) || ismember('pred_med',error_type)
        % ɾ��Med��RAE���е��쳣ֵ����ϵ�������̬
        parmhat = lognfit_outliers(MedianTest(i,:),70); 
        med_params(i,:) = parmhat;
        % ����ģֵ̬
        PEMF_Error_med_mod(i)=exp(parmhat(1)-(parmhat(2))^2);
        % �����׼��
        PEMF_Error_med_var(i)=sqrt((exp((parmhat(2))^2)-1)*exp(2*parmhat(1)+(parmhat(2))^2));
    end
    % MOD-Max,VAR-MAX
    if ismember('mod_max',error_type) || ismember('var_max',error_type) || ismember('pred_max',error_type)
        % ɾ��Max(RAE)�е��쳣ֵ����ϵ�������̬
        parmhat = lognfit_outliers(MaxTest(i,:),60); 
        max_params(i,:) = parmhat;
        % Mode of Max Estimation
        PEMF_Error_max_mod(i)=exp(parmhat(1)-(parmhat(2))^2);
        % �����׼��
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

%% ѡ������������ģ�Ͳ�Ԥ�������
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

%%  �����ֵģֵ̬
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

%%  ������ֵģֵ̬
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

%%  �����ֵ��׼��
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

%%  ������ֵ��׼��
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

% %% ��ͼ
% PEMF_Error_med_mod=[PEMF_Error_med_mod;PEMF_MedError_return];
% PEMF_Error_max_mod=[PEMF_Error_max_mod;PEMF_MaxError_return];
% PEMF_Error_med_var=[PEMF_Error_med_var;PEMF_MedErrorv_return];
% PEMF_Error_max_var=[PEMF_Error_max_var;PEMF_MaxErrorv_return];
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% if(ismember('mod_med',error_type))
%     fig = plot_pemf(x_med,PEMF_Error_med_mod,med_params,model_type_med,v_med);
%     set(get(get(fig,'CurrentAxes'),'Title'),'String','�����ֵ(Modal)')
% end
% if(ismember('mod_max',error_type))
%     fig = plot_pemf(x_max,PEMF_Error_max_mod,max_params,model_type_max,v_max);
%     set(get(get(fig,'CurrentAxes'),'Title'),'String','������ֵ(Modal)')
% end
% if(ismember('var_med',error_type))
%     fig = plot_pemf(x_medv,PEMF_Error_med_var,med_params,model_type_medv,v_medv);
%     set(get(get(fig,'CurrentAxes'),'Title'),'String','�����ֵ(Standard Deviation)')
% end
% if(ismember('var_max',error_type))
%     fig = plot_pemf(x_maxv,PEMF_Error_max_var,max_params,model_type_maxv,v_maxv);
%     set(get(get(fig,'CurrentAxes'),'Title'),'String','������ֵ(Standard Deviation)')
% end
% %%%%%%%%%%%%%%%%%%%%%%%%%%
%% ��������Ҫ�����
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
% ������ʽ���
function [model] = check_input(trainer,X,Y,error_type,n_pnts_step,n_steps,n_pnts)
% ���X��Y�Ƿ񳤶�ƥ��
if(n_pnts ~= size(Y,1))
    error('�뱣֤X�����Y����������ͬ��\n');
end

% ������������Ƿ��㹻
if(n_pnts < n_steps*n_pnts_step + 3)
    error('�����������٣����������������С����������������������ٲ�����\n');
end

% % ���trainer�Ƿ�Ϊ�������
if(~isa(trainer,'function_handle'))
    error('trainer��Ϊ���������');
end

% ����trainer�Ƿ��ܱ����ã��Լ��Ƿ񷵻���ȷ��ʽ
try
    model = trainer(X,Y);
catch ME
    msg = 'trainer��ʽ��ƥ�䣬�޷���ȷ�����á�\n';
    causeException = MException('MATLAB:myCode:dimensions',msg);
    ME = addCause(ME,causeException);
    rethrow(ME)
end

% ���trainer����Ƿ����y = model(x)
try
    model(X(1,:));
catch ME
    msg = 'trainer�����ʽ��ƥ�䣬�޷���ȷ�����á�\n';
    causeException = MException('MATLAB:myCode:dimensions',msg);
    ME = addCause(ME,causeException);
    rethrow(ME)
end

% �����������б���
if length(error_type) < 2
    warning('��ѡ���������͹��١�\n');
end

% �����������Ƿ�Ϊ'mod_med','mod_max','var_med'��'var_max'
for error_n = error_type
    if(~(strcmp(error_n,'mod_med') || strcmp(error_n,'mod_max') || ...
        strcmp(error_n,'var_med') || strcmp(error_n,'var_max') || ...
        strcmp(error_n,'pred_med') || strcmp(error_n,'pred_max')))
        error('���������ҪΪ''mod_med'',''mod_max'',''var_med'',''var_max'',''pred_med''��''pred_max''��\n');
    end
end

% ���Y�Ƿ�Ϊ��������
if(size(Y,2) ~= 1)
    error('YӦ��Ϊ�������ݡ�\n');
end

% �����ݿ�ȹ���ʱ����
if( min(Y)/max(Y) < 5*10^-3)
    warning('���ݿ��ԶԶ����2�������������ܶ����������������ѡ�\n');
end

end

% ɾ���쳣ֵ����ϵ�������̬
function parmhat = lognfit_outliers(data,outlier_percent)
    % ɾ���ٷ�λ�����ϵ��쳣ֵ
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

    % ����ƽ��ֵ����׼��
    CP = CT';
    mu = mean (CP);
    sigma = std (CP);
    [n,~] = size(CP);
    Meanmat = repmat(mu,n,1);
    Sigmamat = repmat(sigma,n,1);
    % ɾ��3������������ĵ�
    outliers = abs(CP-Meanmat) > 3*Sigmamat;
    CP(any(outliers,2),:) = [];
    % ɾ�����
    CP(find(CP==0))=[];
    % ������̬��ϣ����ؾ�ֵ�ͱ�׼��
    parmhat = lognfit(CP(:)); 
end

% ѡ��ع�ģ��
function [RMSE,ErrorPrediction,VCoe] = SelectRegression(X,Y,iSType,NinP,alfa)
% Helper function with a few modes
% - Able to fit PEMF error to two different regression models depending on
%       the value of isType 
% - Able to predict the next value in the regression (ErrorPrediction)
% - Able to give the model fit parameters (VCoe = [a,b])
X(isnan(Y(:,1)),:)=[];
Y(isnan(Y(:,1)),:)=[];

%% 1. ָ���ع麯�� Exponential Fit Model   Y=a*exp(b*X)
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
    
    % ����RMSE
    RMSE=Grmse(data,estimate');
    VCoe=[a11,b11];
    
end
%% 2. �˷��ع麯�� Power Fit Model   Y=a*X^b

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
     
	% ����RMSE
    RMSE=Grmse(data,estimate');
    VCoe=[a11,b11];
    
end
end 

% �˷��ع�Power Fit Model
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

% ����RMSE
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

% ƽ���ȱ�׼�����ϵ����
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

% ��ͼ
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
legend([dist,calc,pred],'���ֲ�','�м��������ֵ','���մ�������ֵ');
title('PEMF Error'); ylabel('���������ֵ��');
if(Y(end) == Y(end-1))
    n_points = X(end) + step;
else
    n_points = X(end);
end
s = strcat({'ѵ������(��ѵ������Ϊ'},num2str(n_points),{')'});
xlabel(s{1});
     
end