work_dir="D:\DRLresults\SB3";
cd(work_dir);
filename = '22_debug.csv';
start_col = 2;
[results_output,var_name] = readcsv(filename,start_col);

[Drag_DT,Drag_STD] = ProcessOutput(results_output);

function [Drag_DT,Drag_STD] = ProcessOutput(data)
%%% This function is used for analyzing the output data from training phase
%%% by RL. The target is to show the details of training during each
%%% episode and find a convergence criteria for training
%%% The output of the function is the detrended drag data during training
%%% and the std of detrended drag for each episode.
% work_dir="D:\DRLresults\SB3\SAC_FM0FS_tri512";
% cd(work_dir);
% filename = '11_debug.csv';
% start_col = 2;
% [results_output,var_name] = readcsv(filename,start_col);
Episode = data(:,1);
Step = data(:,2); %% This is the numerical step
RecArea = data(:,3);
Drag = -2*data(:,4);
Lift = data(:,5);
Jet = abs(data(:,6)); 

Epi_length = 50000; % There are 5000 samples for each episode now
Epi_num = Episode(end)-1; % The number of episodes we want to analyze. The last episode may not be full

%% Plot for the whole training period
Total_Step = 10*Epi_length*(Episode-1) + Step;

subplot(2,2,1)
plot(Total_Step,Drag)
title(['Cd during all learning episodes']);
xlabel('Numerical steps')
ylabel('Cd')
%% Reshape the data
Drag_Reshape = zeros(Epi_length,Epi_num);
Step_Reshape = zeros(Epi_length,Epi_num);
for i = 1 : Epi_num
    Drag_Reshape(:,i) = Drag(((i-1)*Epi_length+1):i*Epi_length);
    Step_Reshape(:,i) = Step(((i-1)*Epi_length+1):i*Epi_length);
%     if i == Episode(end)
%         Drag_Reshape(:,i) = Drag(((i-1)*5000+1):end);
%     else
%         Drag_Reshape(:,i) = Drag(((i-1)*5000+1):i*5000);
%     end
end

%% Plot for specific episode
spec_epi = Epi_num-2; % Notice: Not more than Epi_num
spec_step = Step_Reshape(:,spec_epi);
Drag_spec = Drag_Reshape(:,spec_epi);
subplot(2,2,2)
plot(spec_step,Drag_spec)
title(['Cd for a specific learning episode (No.',num2str(spec_epi),')']);
xlabel('Numerical steps')
ylabel('Cd')
% spec_step = 10*(((spec_epi-1)*Epi_length+1):spec_epi*Epi_length);
% Drag_spec = Drag(spec_step/10);


%% Calculate convergence details for each episode
poly_detrend = 9; % The order of polynomial to detrend the data
Drag_DT = detrend(Drag_Reshape,poly_detrend); % Detrend the data
subplot(2,2,3)
for i = 15:Epi_num
plot(Step_Reshape(:,i),Drag_DT(:,i))
hold on
end
title(['Detrended Cd for each episode with polynomial order ',num2str(poly_detrend)]);
xlabel('Numerical steps')
ylabel('Cd (Detrended)')
Drag_DT_trunc = Drag_DT(end/2:end,:);
Drag_STD = std(Drag_DT_trunc);
if size(Drag_STD,2) == Epi_num
    subplot(2,2,4)
    x = 1:Epi_num;
    plot(x,Drag_STD)
    title(['STD of detrended Cd (from last half episode)']);
    xlabel('Number of episodes')
    ylabel('std(Cd)')
else
    print("Error. Dimension of std results doesn't match the number of episodes.")

end

end



%%%%%%%%%%
% The function used to read data from csv files
function [results,var_name] = readcsv(filename,start_col)
%% start_col specifies which column to start to read in the file
if isfile(filename)
    opts = detectImportOptions(filename);
    prev = preview(filename,opts);
    % Parameters for reading (check before reading)
    dim=size(prev);
    opts.SelectedVariableNames = [start_col:dim(2)];
    opts.DataLines = 2;
    % Store data in a matrix
    results = readmatrix(filename,opts);
    var_name = opts.VariableNames;
else
    disp("No csv file is found.")
    results=0;
    var_name=0;
end
end