% Multiobjective Marine Predators Algorithm (Developed in MATLAB R2017b)
% 
% ---Beijing Institute of Technology 
%_________________________________________________________________________
%  Marine Predators Algorithm source code (Developed in MATLAB R2015a)
%
%  programming: Afshin Faramarzi & Seyedali Mirjalili
%
% paper:
%  A. Faramarzi, M. Heidarinejad, S. Mirjalili, A.H. Gandomi, 
%  Marine Predators Algorithm: A Nature-inspired Metaheuristic
%  Expert Systems with Applications
%  DOI: doi.org/10.1016/j.eswa.2020.113377
%_________________________________________________________________________

function [POS_fit,POS]=IMOMPA2(SearchAgents_no,Max_iter,lb,ub,dim,fobj,nobj,nInt)

stepsize=zeros(SearchAgents_no,dim);
fitness=inf(SearchAgents_no,nobj);
archive_size=floor(SearchAgents_no*0.8);
POS_fit=[];
POS=[];

Prey=initialization(SearchAgents_no,dim,ub,lb);

Xmin=repmat(ones(1,dim).*lb,SearchAgents_no,1);
Xmax=repmat(ones(1,dim).*ub,SearchAgents_no,1);
         
Iter=0;
FADs=0.2;
P=0.5;

mid=(lb+ub)/2;

while Iter<Max_iter    
     %------------------- Detecting top predator -----------------    
 for i=1:SearchAgents_no  
     
    Flag4ub=Prey(i,:)>ub;
    Flag4lb=Prey(i,:)<lb;    
    Prey(i,:)=(Prey(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
    Prey(:,nInt) = round(Prey(:,nInt));
    fitness(i,:)=fobj(Prey(i,:));
            
 end
 
 % 计算准对立解
 if rand()<0.8
    for i=1:size(Prey,1) 
        Prey_oppo=mid+(mid-Prey(i,:)).*rand(1,dim);
        Prey_oppo(:,nInt) = round(Prey_oppo(:,nInt));
        fitness_oppo=fobj(Prey_oppo);
        if dominates(fitness_oppo,fitness(i,:))
            Prey(i,:)=Prey_oppo;
            fitness(i,:)=fitness_oppo;
        end
    end
 end 
 
 % find non-dominated solutions
 DOMINATED = checkDomination(fitness);
 temp_fit = [fitness(~DOMINATED,:);POS_fit];
 temp_pos = [Prey(~DOMINATED,:);POS];
 if size(temp_fit,1) > 2 && Iter > 0
     DOMINATED = checkDomination(temp_fit);
     POS_fit = temp_fit(~DOMINATED,:);
     POS = temp_pos(~DOMINATED,:);
 else
     POS_fit = temp_fit;
     POS = temp_pos;
 end

 if size(POS,1) > 1
    [POS, NON_DOMINATED,~] = unique(POS,'rows');
    POS_fit = POS_fit(NON_DOMINATED,:);
 end
 
 % delete most crowded solutions when the archive is full
 if size(POS_fit,1) > archive_size
     pop_index = crowdedDistances(POS_fit,size(POS_fit,1)-archive_size);
     POS_fit(pop_index,:) = [];
     POS(pop_index,:) = [];
 end
 
 % construct Elite matrix
 Elite=repmat(POS,ceil(SearchAgents_no/size(POS,1)),1);
 Elite(SearchAgents_no+1:end,:)=[];
     
     %------------------- Marine Memory saving ------------------- 
    
 if Iter==0
   fit_old=fitness;    Prey_old=Prey;
 end
 
  [Inx,NumD]=checkDomination([fitness;fit_old]);
  Inx=Inx(1:SearchAgents_no,:);
  Indx=repmat(Inx,1,dim);
  Prey=Indx.*Prey_old+~Indx.*Prey;
  fitness=Inx.*fit_old+~Inx.*fitness;
  NumD=Inx.*NumD(SearchAgents_no+1:end,:)+~Inx.*NumD(1:SearchAgents_no,:);
        
  fit_old=fitness;    Prey_old=Prey;

     %------------------------------------------------------------   
     
%  Elite=repmat(Top_predator_pos,SearchAgents_no,1);  %(Eq. 10) 
 CF=(1-Iter/Max_iter)^(2*Iter/Max_iter);
                             
 RL=0.05*levy(SearchAgents_no,dim,1.5);   %Levy random number vector
 RB=randn(SearchAgents_no,dim);          %Brownian random number vector
 
 % 策略分界支配数
 lb_NumD=min(NumD);
 ub_NumD=max(NumD);
 lb_iter=(ub_NumD-lb_NumD)/3+lb_NumD;
 ub_iter=2*(ub_NumD-lb_NumD)/3+lb_NumD;
 mid_iter=(ub_NumD+lb_NumD)/2;
 
 for i=1:SearchAgents_no
     for j=1:dim        
       R=rand();
          %------------------ Phase 1 (Eq.12) ------------------- 
       if NumD(i)<lb_iter
           stepsize(i,j)=RB(i,j)*(Elite(i,j)-RB(i,j)*Prey(i,j));                    
           Prey(i,j)=Prey(i,j)+P*R*stepsize(i,j); 
             
          %--------------- Phase 2 (Eqs. 13 & 14)----------------
       elseif NumD(i)>lb_iter && NumD(i)<mid_iter
           stepsize(i,j)=RB(i,j)*(RB(i,j)*Elite(i,j)-Prey(i,j));
           Prey(i,j)=Elite(i,j)+P*CF*stepsize(i,j);  
         
       elseif NumD(i)>mid_iter && NumD(i)<ub_iter
           stepsize(i,j)=RL(i,j)*(Elite(i,j)-RL(i,j)*Prey(i,j));                     
           Prey(i,j)=Prey(i,j)+P*R*stepsize(i,j);  
           
         %----------------- Phase 3 (Eq. 15)-------------------
       else 
           stepsize(i,j)=RL(i,j)*(RL(i,j)*Elite(i,j)-Prey(i,j)); 
           Prey(i,j)=Elite(i,j)+P*CF*stepsize(i,j);  
    
       end  
      end                                         
  end    
        

       %------------------- Detecting top predator -----------------    
 for i=1:SearchAgents_no  
        
    Flag4ub=Prey(i,:)>ub;
    Flag4lb=Prey(i,:)<lb;    
    Prey(i,:)=(Prey(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
    Prey(:,nInt) = round(Prey(:,nInt));
    fitness(i,:)=fobj(Prey(i,:));
        
 end
 
 % find non-dominated solutions
 DOMINATED = checkDomination(fitness);
 temp_fit = [fitness(~DOMINATED,:);POS_fit];
 temp_pos = [Prey(~DOMINATED,:);POS];
 if size(temp_fit,1) > 2 && Iter > 0
     DOMINATED = checkDomination(temp_fit);
     POS_fit = temp_fit(~DOMINATED,:);
     POS = temp_pos(~DOMINATED,:);
 else
     POS_fit = temp_fit;
     POS = temp_pos;
 end

 if size(POS,1) > 1
    [POS, NON_DOMINATED,~] = unique(POS,'rows');
    POS_fit = POS_fit(NON_DOMINATED,:);
 end
 
 % delete most crowded solutions when the archive is full
 if size(POS_fit,1) > archive_size
     pop_index = crowdedDistances(POS_fit,size(POS_fit,1)-archive_size);
     POS_fit(pop_index,:) = [];
     POS(pop_index,:) = [];
 end
        
     %---------------------- Marine Memory saving ----------------
    
 if Iter==0
    fit_old=fitness;    Prey_old=Prey;
 end
     
    [Inx,~]=checkDomination([fitness;fit_old]);
    Inx=Inx(1:SearchAgents_no,:);
    Indx=repmat(Inx,1,dim);
    Prey=Indx.*Prey_old+~Indx.*Prey;
    fitness=Inx.*fit_old+~Inx.*fitness;
        
    fit_old=fitness;    Prey_old=Prey;
    
%  % plotting
%  if Iter==0
%     if(size(POS_fit,2) == 2)
%         h_fig = figure();
%         h_par = scatter(POS_fit(:,1),POS_fit(:,2),20,'filled', 'markerFaceAlpha',0.3,'MarkerFaceColor',[128 193 219]./255); hold on;
%         h_rep = plot(POS_fit(:,1),POS_fit(:,2),'ok'); hold on;
%         grid on; xlabel('f1'); ylabel('f2');
%         drawnow;
%         axis square;
%     elseif(size(POS_fit,2) == 3)
%         h_fig = figure();
%         h_rep = plot3(POS_fit(:,1),POS_fit(:,2),POS_fit(:,3),'ok'); hold on;
%         grid on; xlabel('f1'); ylabel('f2'); zlabel('f3');
%         drawnow;
%         axis square;
%     end
%  else
%      if(size(POS_fit,2) == 2)
%         figure(h_fig); delete(h_rep);
%         h_par = scatter(POS_fit(:,1),POS_fit(:,2),20,'filled', 'markerFaceAlpha',0.3,'MarkerFaceColor',[128 193 219]./255); hold on;
%         h_rep = plot(POS_fit(:,1),POS_fit(:,2),'ok'); hold on;
%         grid on; xlabel('f1'); ylabel('f2');
%         drawnow;
%         axis square;
%      elseif(size(POS_fit,2) == 3)
%         figure(h_fig); delete(h_rep);
%         h_rep = plot3(POS_fit(:,1),POS_fit(:,2),POS_fit(:,3),'ok'); hold on;
%         grid on; xlabel('f1'); ylabel('f2');
%         drawnow;
%         axis square;
%      end
%  end
 
     %---------- Eddy formation and FADs? effect (Eq 16) ----------- 
                             
  if rand()<FADs
     U=rand(SearchAgents_no,dim)<FADs;                                                                                              
     Prey=Prey+CF*((Xmin+rand(SearchAgents_no,dim).*(Xmax-Xmin)).*U);

  else
     r=rand();  Rs=SearchAgents_no;
     stepsize=(FADs*(1-r)+r)*(Prey(randperm(Rs),:)-Prey(randperm(Rs),:));
     Prey=Prey+stepsize;
  end
                                                        
  Iter=Iter+1;  
       
end
end

% This function initialize the first population of search agents
function Positions=initialization(SearchAgents_no,dim,ub,lb)
% Tent映射
Positions=rand(1,dim);
for i=2:SearchAgents_no
    Positions(i,:)=mod(2*Positions(i-1,:),1)+rand(1,dim)/SearchAgents_no;
end

Positions=Positions.*(ub-lb)+lb;
end

function [z] = levy(n,m,beta)
% used for Numerator 
num = gamma(1+beta)*sin(pi*beta/2); 

% used for Denominator
den = gamma((1+beta)/2)*beta*2^((beta-1)/2); 

% Standard deviation
sigma_u = (num/den)^(1/beta);

u = random('Normal',0,sigma_u,n,m); 
v = random('Normal',0,1,n,m);
z =u./(abs(v).^(1/beta));
end

function [dom_vector,dom_num] = checkDomination(fitness)
Np = size(fitness,1);
dom_vector = zeros(Np,1);
dom_num = zeros(Np,1);
all_perm = nchoosek(1:Np,2);    % Possible permutations
all_perm = [all_perm; [all_perm(:,2) all_perm(:,1)]];

d = dominates(fitness(all_perm(:,1),:),fitness(all_perm(:,2),:));
dominated_particles = unique(all_perm(d==1,2));
dom_vector(dominated_particles) = 1;

dominated_paar = all_perm(d==1,:);

for i=1:Np
    dom_num(i,1) = sum(dominated_paar(:,1)==i);
end
end

% Function that returns 1 if x dominates y and 0 otherwise
function d = dominates(x,y)
    d = all(x<=y,2) & any(x<y,2);
end

% Crowded distances
function pop_index = crowdedDistances(fitness,n_pop)
crowd = zeros(size(fitness,1),1);
for i = 1:size(fitness,2)
    fit_max = max(fitness(:,i));
    fit_min = min(fitness(:,i));
    for j = 2:length(crowd)-1
        crowd(j) = crowd(j) + (abs(fitness(j-1,i)-fitness(j+1,i)))./(fit_max-fit_min);
    end
end
crowd(1) = Inf;
crowd(length(crowd)) = Inf;

[~,pop_index] = sort(crowd);
pop_index = pop_index(1:n_pop);
end
