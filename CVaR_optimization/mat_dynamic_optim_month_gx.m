fprintf("CVaR optimization for GX-GRU model with Dow dataset\n");
%% exit with[nohup ./matlab -nosplash -nodisplay -nodesktop < CVaR_optimization/mat_dynamic_optim_month_gx.m > data/cvar_ouput/result_gx_month_time_att.log 2>&1 &]

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%dataset read in %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf("file_start_to_read_in... \n");

%% 设置导入选项并导入数据
opts = delimitedTextImportOptions("NumVariables", 29);

% 指定范围和分隔符
opts.DataLines = [1, Inf];
opts.Delimiter = ",";

% 指定列名称和类型
opts.VariableNames = ["VarName1", "VarName2", "VarName3", "VarName4", "VarName5", "VarName6", "VarName7", "VarName8", "VarName9", "VarName10", "VarName11", "VarName12", "VarName13", "VarName14", "VarName15", "VarName16", "VarName17", "VarName18", "VarName19", "VarName20", "VarName21", "VarName22", "VarName23", "VarName24", "VarName25", "VarName26", "VarName27", "VarName28", "VarName29"];
opts.VariableTypes = ["double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double"];

% 指定文件级属性
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% 导入数据
parammatrixSgamma = readtable("data/gru_save/param_matrix_S_gamma_step_one_ahead_month_time_gx_att.csv", opts);

%% 清除临时变量
clear opts

%% 设置导入选项并导入数据
opts = delimitedTextImportOptions("NumVariables", 29);

% 指定范围和分隔符
opts.DataLines = [1, Inf];
opts.Delimiter = ",";

% 指定列名称和类型
opts.VariableNames = ["VarName1", "VarName2", "VarName3", "VarName4", "VarName5", "VarName6", "VarName7", "VarName8", "VarName9", "VarName10", "VarName11", "VarName12", "VarName13", "VarName14", "VarName15", "VarName16", "VarName17", "VarName18", "VarName19", "VarName20", "VarName21", "VarName22", "VarName23", "VarName24", "VarName25", "VarName26", "VarName27", "VarName28", "VarName29"];
opts.VariableTypes = ["double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double"];

% 指定文件级属性
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% 导入数据
parammatrixSbeta = readtable("data/gru_save/param_matrix_S_beta_step_one_ahead_month_time_gx_att.csv", opts);


%% 清除临时变量
clear opts

%% 设置导入选项并导入数据
opts = delimitedTextImportOptions("NumVariables", 29);

% 指定范围和分隔符
opts.DataLines = [1, Inf];
opts.Delimiter = ",";

% 指定列名称和类型
opts.VariableNames = ["VarName1", "VarName2", "VarName3", "VarName4", "VarName5", "VarName6", "VarName7", "VarName8", "VarName9", "VarName10", "VarName11", "VarName12", "VarName13", "VarName14", "VarName15", "VarName16", "VarName17", "VarName18", "VarName19", "VarName20", "VarName21", "VarName22", "VarName23", "VarName24", "VarName25", "VarName26", "VarName27", "VarName28", "VarName29"];
opts.VariableTypes = ["double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double"];

% 指定文件级属性
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% 导入数据
parammatrixSalpha = readtable("data/gru_save/param_matrix_S_alpha_step_one_ahead_month_time_gx_att.csv", opts);


%% 清除临时变量
clear opts

%% 设置导入选项并导入数据
opts = delimitedTextImportOptions("NumVariables", 29);

% 指定范围和分隔符
opts.DataLines = [1, Inf];
opts.Delimiter = ",";

% 指定列名称和类型
opts.VariableNames = ["VarName1", "VarName2", "VarName3", "VarName4", "VarName5", "VarName6", "VarName7", "VarName8", "VarName9", "VarName10", "VarName11", "VarName12", "VarName13", "VarName14", "VarName15", "VarName16", "VarName17", "VarName18", "VarName19", "VarName20", "VarName21", "VarName22", "VarName23", "VarName24", "VarName25", "VarName26", "VarName27", "VarName28", "VarName29"];
opts.VariableTypes = ["double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double"];

% 指定文件级属性
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% 导入数据
predYmatrix = readtable("data/gru_save/predY_matrix_month.csv", opts);


%% 清除临时变量
clear opts


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
%% 设置导入选项并导入数据
opts = delimitedTextImportOptions("NumVariables", 5);

% 指定范围和分隔符
opts.DataLines = [1, Inf];
opts.Delimiter = ",";

% 指定列名称和类型
opts.VariableNames = ["VarName1", "VarName2", "VarName3", "VarName4", "VarName5"];
opts.VariableTypes = ["double", "double", "double", "double", "double"];

% 指定文件级属性
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% 导入数据
UVmatrixgx = readtable("data/gru_save/UV_matrix_gx_month_time_att.csv", opts);

%% 清除临时变量
clear opts


fprintf("file_read_in_successfully \n");


rng(817);
warning('off');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%
parammatrixSalpha = parammatrixSalpha{2:end,2:end};
parammatrixSbeta = parammatrixSbeta{2:end,2:end};
parammatrixSgamma = parammatrixSgamma{2:end,2:end};
UV_matrix_gx = UVmatrixgx{2:end,2:end};
testset = predYmatrix{2:end,2:end};
K = length(parammatrixSalpha(1,:));
T = length(parammatrixSalpha(:,1));

tl1= 1+1*(2611+652-1)+200;
interval = 21;
TK = T;
B = 10;
beta_M_seq=[0.90,0.95,0.99;5000,10000,50000];
R_seq=(1:3)*1e-2;
R_seq = [R_seq 0];
lowbound=0;

gz =@(x,u,v) x.*((u.^(x)+v.^(-x))/4+1);

return_mat = ones(T,length(R_seq)*B*length(beta_M_seq(1,:)));
naive_mat = ones(T,1);
for j = 1:TK
    j1 = (j-1)*interval+1;
    j2 = j*interval;
    Salpha = parammatrixSalpha(j,:);
    Sbeta = parammatrixSbeta(j,:);
    Sgamma = diag((parammatrixSgamma(j,:)));
    testset_j = testset(j,:);
    naive_mat(j,:) = testset_j*ones(K,1)/K;
    R_seq(length(R_seq)) = naive_mat(j,:);
    for i_m = 1:length(beta_M_seq(1,:))
        M = beta_M_seq(2,i_m);
        beta = beta_M_seq(1,i_m);
        Z=normrnd(0,1,[M*B,K+1]);
        YMat = ones(M*B,1)*Salpha;
        for kk =1:K
            uuM = UV_matrix_gx(kk+1,1);
            vvM = UV_matrix_gx(kk+1,2);
            uui = UV_matrix_gx(kk+1,3);
            vvi = UV_matrix_gx(kk+1,4);
            Zm_g = gz(Z(:,1),uuM,vvM);
            Zi_g = gz(Z(:,kk+1),uui,vvi);
            YMat(:,kk) = YMat(:,kk)+ Sbeta(kk)*Zm_g +Sgamma(kk)*Zi_g;
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

            return_mat(j,k3:k4)=testset_j*weight;

        catch
            %flag=1;
            %break;
        end
        %%%%
    end
    %%%
    end
    if mod(j,10)==0
        fprintf('%d th period is finished! \n',j);
    end
    %%%%
end
writematrix(return_mat,'data/cvar_ouput/gx_cumulation.csv');

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
