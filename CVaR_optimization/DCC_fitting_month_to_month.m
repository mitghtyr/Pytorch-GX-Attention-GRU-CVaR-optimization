fprintf("DCC-MM model training and CVaR optimization. \n");
%% exit with[nohup ./matlab -nosplash -nodisplay -nodesktop < CVaR_optimization/DCC_fitting_month_to_month.m > data/cvar_ouput/result_dcc_mm.log 2>&1 &]

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%dataset read in %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
opts = delimitedTextImportOptions("NumVariables", 30);

% 指定范围和分隔符
opts.DataLines = [2, Inf];
opts.Delimiter = ",";

% 指定列名称和类型
opts.VariableNames = ["Date", "GSPC", "AXP", "XOM", "AAPL", "BA", "CAT", "CSCO", "CVX", "GS", "HD", "PFE", "IBM", "INTC", "JNJ", "KO", "JPM", "MCD", "MMM", "MRK", "MSFT", "NKE", "PG", "TRV", "UNH", "RTX", "VZ", "WBA", "WMT", "DIS"];
opts.VariableTypes = ["datetime", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double"];

% 指定文件级属性
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% 指定变量属性
opts = setvaropts(opts, "Date", "InputFormat", "yyyy-MM-dd");

% 导入数据
dailyreturn = readtable("data/daily_return_ratio.csv", opts);

%% 清除临时变量
clear opts


fprintf("file_read_in_successfully \n");
addpath('~/.jupyter/optimization');
dirList = {'bootstrap',...
    'crosssection',...
    'distributions',...
    'GUI',...
    'multivariate',...
    'tests',...
    'timeseries',...
    'univariate',...
    'utility',...
    'realized',...
    };


for i=1:length(dirList)
    addpath(fullfile('mfe-toolbox-master',dirList{i}));
end

rng(817);
warning('off');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tl1= 1+200+2601+ 650+21;
tl2 = 103*21;

testset = dailyreturn{tl1:(tl1+tl2-1),2:end};
%% input with 28+1 stocks
hist_mat = dailyreturn{1:tl1-1,2:end};
whole_mat = dailyreturn{1:(tl1+tl2-1),2:end};

K = length(testset(1,:))-1;
T = length(testset(:,1));
interval = 21;
TK = floor(T/interval);
TK1 = floor((tl1-1)/interval);
B = 10;
beta_M_seq=[0.90,0.95,0.99;5000,10000,50000];
R_seq=(1:3)*1e-2;
R_seq = [R_seq 0];
lowbound=0;


hist_mat_month = zeros(TK1,K+1);
for i=1:TK1
    id1 = tl1-1-TK1*interval+(i-1)*(interval)+1;
    id2 = id1+interval-1;
    r_seq = prod(hist_mat(id1:id2,:)+1,1)-1;
    hist_mat_month(i,:)= r_seq;

end
mean_all = mean(hist_mat_month,1);
std_all = std(hist_mat_month,1);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%DCC-MM training%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sigma_M_mat = ones(TK+1,1);
%% for Market return fit singkle Garch
YMt = (hist_mat_month(:,1)-mean_all(1))/std_all(1);

[parameters_M, ll_M ,Ht_M]=tarch(YMt,1,0,1);
sigma_M_start = parameters_M(1)+ parameters_M(2)*(YMt(end))^2 + parameters_M(3)*Ht_M(end);
sigma_M_mat(1) = sqrt(sigma_M_start );

%% for individual stocks fit 2-DCC with the market return
% fit 1-order DCC
parammatrix = ones(6,K);


theta_start = ones(2,K);
%% ( sigma_i, rho,2*K)
rho_mat = ones(TK+1,K);

for kk=1:K
    YMt = (hist_mat_month(:,1)-mean_all(1))/std_all(1);
    Yit = (hist_mat_month(:,kk+1)-mean_all(kk+1))/std_all(kk+1);
    [parameters, ll ,Ht, VCV]=dcc([YMt,Yit],[],1,0,1);
    parammatrix(:,kk) = parameters(4:end);
    theta_start(1,kk) = parameters(4)+parameters(5)* (Yit(end))^2+ parameters(6)* Ht(2,2,end);

    Q_t =ones(2,2);
    Q_t(1,2) = Ht(1,2,end)/sqrt(Ht(1,1,end)*Ht(2,2,end));
    Q_t(2,1) = Q_t(1,2);
    R_hat = ones(2,2);
    R_hat(1,2) = parameters(7);
    R_hat(2,1) = R_hat(1,2);
    Y_t=ones(2,1);
    Y_t(1) =YMt(end)/sqrt(Ht(1,1,end));
    Y_t(2) =Yit(end)/sqrt(Ht(2,2,end));
    Q_pred =(1-parameters(8)-parameters(9))* R_hat+parameters(8)*(Y_t*Y_t')+parameters(9)*Q_t;
     %%'
    rho_t = Q_pred(1,2)/sqrt(Q_pred(1,1)*Q_pred(2,2));
    theta_start(2,kk) = rho_t;
    rho_mat(1,kk) = rho_t;


end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%CVaR optimization%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
return_mat = ones(TK,length(R_seq)*B*length(beta_M_seq(1,:)));
naive_mat = ones(TK,1);
YMat_TK = zeros(10000*10,TK);

for j = 1:TK
    j1 = (j-1)*interval+1;
    j2 = j*interval;

    testset_j = testset(j1:j2,:);
    monthly_true_return = prod(testset_j(:,2:end)+1,1)-1;

    naive_mat(j,:) = monthly_true_return*ones(K,1)/K;
    R_seq(length(R_seq)) = naive_mat(j,:);


 for i_m = 1:length(beta_M_seq(1,:))
    M = beta_M_seq(2,i_m);
    beta = beta_M_seq(1,i_m);
    %%Z=normrnd(0,1,[M*B,K+1]);

    ZM=normrnd(0,1,[M*B,1]);
    sigma_M_month = ones(M*B,1);
    sigma_M_month(:,1) = sigma_M_month(:,1)*sigma_M_start;

    YMat = zeros(M*B,K);
    for kk =1:K
      sigma_S_month = ones(M*B,1);
      sigma_S_month(:,1) = sigma_S_month(:,1)*theta_start(1,kk);

      rho_S_month = ones(M*B,1);
      rho_S_month(:,1) = rho_S_month(:,1)*theta_start(2,kk);
      Zi=normrnd(0,1,[M*B,1]);
      mon_temp = ones(M*B,1);
      mon_temp(:,1) =  sqrt(theta_start(1,kk))*(ZM(:,1) *theta_start(2,kk)+Zi(:,1) *sqrt(1-theta_start(2,kk)*theta_start(2,kk)));


      mon_temp = mon_temp*std_all(kk+1)+mean_all(kk+1);
      % mu is the mean of past 200 days and sigma is predicted

      YMat(:,kk) =  mon_temp;

    end
    if (M==10000)
        YMat_TK(:,j) = mean(YMat,2);
    end
  for b=1:B
    i1 = (b-1)*M+1;
    i2 = b*M;

    p = PortfolioCVaR('NumAssets', K ,'LowerBound', lowbound,'UpperBound',1-lowbound, 'Budget', 1);

    try
       p = setScenarios(p, YMat(i1:i2,:));

       p = setProbabilityLevel(p, beta);
       weight = estimateFrontierByReturn(p, R_seq);
       k3=(i_m-1)*B*length(R_seq)+(b-1)*length(R_seq)+1;
       k4=(i_m-1)*B*length(R_seq)+(b)*length(R_seq);

       return_mat(j,k3:k4)=monthly_true_return*weight;

    catch
       %flag=1;
       %break;
    end
    %% try end
  end
  %%
 end
 %%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%update parameters%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

     YMt = prod(testset_j(:,1)+1,1)-1;
     YMt = (YMt-mean_all(1))/std_all(1);

     eMt = YMt/sqrt(sigma_M_start);
     sigma_M_start = parameters_M(1)+(parameters_M(2)* (YMt)^2+ parameters_M(3))* sigma_M_start;
     sigma_M_mat(j+1) = sqrt(sigma_M_start);
     for kk= 1:K
        Yit = (monthly_true_return (kk)-mean_all(kk+1))/std_all(kk+1);
        eit = Yit/sqrt(theta_start(1,kk));
        theta_start(1,kk) = parammatrix(1,kk)+parammatrix(2,kk)* (Yit )^2+ parammatrix(3,kk)* theta_start(1,kk) ;


        Q_pred =(1-parammatrix(5,kk)-parammatrix(6,kk))* parammatrix(4,kk)+parammatrix(5,kk)*(eMt*eit)+parammatrix(6,kk)*theta_start(2,kk);
        Q_pred = Q_pred./sqrt(1-parammatrix(5,kk)+parammatrix(5,kk)*(eMt*eMt));
        Q_pred = Q_pred./sqrt(1-parammatrix(5,kk)+parammatrix(5,kk)*(eit*eit));
        theta_start(2,kk) = Q_pred;
        rho_mat(j+1,kk) = Q_pred;
     end
     %%%%%%%

   if mod(j,10)==0
        fprintf('%d th period is finished! \n',j);
   end
   %%%%%%%%
end
%%
rho_mat=rho_mat(1:TK,:);
sigma_M_mat = std_all(1)*sigma_M_mat(1:TK);
writematrix(return_mat,'data/cvar_ouput/dcc_dm_cumulation.csv');
writematrix(YMat_TK,'data/cvar_ouput/Simu_return_mat_dcc_dynamic_month_to_month.csv');
writematrix(rho_mat,'data/cvar_ouput/dcc_rho_month.csv');
writematrix(sigma_M_mat,'data/cvar_ouput/dcc_market_sigma.csv');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%evaluation%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
eval_seq=["AV";"SD";"IR";"Drawdown";"CVAR-0.95";"Skewness";"CVaR-ratio";"Rachev"];

NaiveN=zeros(length(eval_seq),1);
NaiveN(1)=mean(naive_mat)*12;
NaiveN(2)=sqrt(12*var(naive_mat));
NaiveN(3)=NaiveN(1)/NaiveN(2);
NaiveN(4)=maxdrawdown(cumprod(naive_mat+1));

NaiveN(5)=12*mean(naive_mat(naive_mat<=quantile(naive_mat,0.05)));
NaiveN(6)=skewness(naive_mat);
NaiveN(7)=NaiveN(1)/abs(NaiveN(5));
upper_es = 12*mean(naive_mat(naive_mat>quantile(naive_mat,0.05)));
NaiveN(8) = upper_es/abs(NaiveN(5));

fmt=['%s:' repmat(' %.4f ',1,1) '\n'];
fprintf(fmt,[eval_seq NaiveN]');

predicted_return=num2cell(return_mat,1);
predicted_return=cellfun(@(x) x(x~=1),predicted_return,'UniformOutput',false);
mat_CVAR=zeros(length(eval_seq),B*length(R_seq)*length(beta_M_seq(1,:)));
mat_CVAR(1,:) = cellfun(@(x) mean(x),predicted_return)*12;
mat_CVAR(2,:)= cellfun(@(x) sqrt(12*var(x)),predicted_return);
mat_CVAR(3,:)= mat_CVAR(1,:)./mat_CVAR(2,:);
mat_CVAR(4,:)=cellfun(@(x) maxdrawdown(cumprod(x+1)),predicted_return);
mat_CVAR(5,:)=cellfun(@(x) 12*mean(x(x<=quantile(x,0.05))),predicted_return);
mat_CVAR(6,:) = cellfun(@(x) skewness(x),predicted_return);
mat_CVAR(7,:)= mat_CVAR(1,:)./abs(mat_CVAR(5,:));
upper_es=cellfun(@(x) 12*mean(x(x>quantile(x,0.05))),predicted_return);
mat_CVAR(8,:)= upper_es./abs(mat_CVAR(5,:));

%id_group=repmat((1:length(R_seq)*length(beta_seq)*length(M_seq))',B,1);
id_group=repmat((1:length(R_seq)*length(beta_M_seq(1,:)))',B,1);
[ii,jj] = ndgrid( id_group, 1:length(eval_seq));
 iijj  = [ii(:), jj(:)];
 Xw=mat_CVAR';
 sums  = accumarray( iijj, Xw(:) ) ;
 cnts  = accumarray( iijj, ones( numel( Xw ), 1 )) ;
 means = sums ./ cnts ;

 sumsq  = accumarray( iijj, (Xw(:)).*(Xw(:))) ;
 sds = sqrt((sumsq-cnts.*means.*means) ./ (cnts-1)) ;

fmt=['%s:' repmat(' %.6f ',1,length(R_seq)-1) '%s' '\n'];

fprintf(fmt,["R " R_seq(1:length(R_seq)-1) "Naive"]');

fmt=['%s:' repmat(' %.4f (%.4f)',1,length(R_seq)) '\n'];

for i_b=1:length(beta_M_seq(1,:))
    fprintf('beta= %f, M= %f : \n',beta_M_seq(1,i_b),beta_M_seq(2,i_b));
        %k1=(i_b-1)*length(M_seq)*length(R_seq)+(i_m-1)*length(R_seq)+1;
        %k2=(i_b-1)*length(M_seq)*length(R_seq)+(i_m)*length(R_seq);
        k1=(i_b-1)*length(R_seq)+1;
        k2=(i_b)*length(R_seq);
    mat_cat=[(means(k1:k2,:))';(sds(k1:k2,:))'];
    mat_cat=reshape(mat_cat(:),length(eval_seq),2*length(R_seq));
    fprintf(fmt,[eval_seq mat_cat]');
    %end
    %===========
end

