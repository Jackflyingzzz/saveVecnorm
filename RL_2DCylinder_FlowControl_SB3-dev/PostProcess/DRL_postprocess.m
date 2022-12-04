% This script is written by Chengwei Xia (Imperial College London), for doing postprocessing to the
% results obtained from the reinforcement learning code SqCyl2DFlowControlDRLParallel.
% The reinforcement learning code is initially developed by Rabault et
% al.(https://github.com/jerabaul29/Cylinder2DFlowControlDRLParallel), and
% later developed by Gonzalo Ancochea Blanco (Imperial College London)
% into rectangular cylinder case.


% % Add path to working directory
% cd 'D:\DRLresults';
% dir
% usr_path=input('Please input the folder or file you want to add to the path.');
% if isfile(usr_path)
%     addpath(genpath(usr_path));
%     
% else
%     
%     quit
% end

%%%%%%%%%%
% Read the results from .csv file (go to the folder manually on the LHS window first)
Color_select=[0.4660 0.6740 0.1880,0.1;
    1,0,0,0.1;
    0,0,1,0.1;
    0.9290 0.6940 0.1250,0.1;
    0.4660 0.6740 0.1880,0.1;
    0.4940 0.1840 0.5560,0.1];
loop=1;
stop=1;
while stop
    work_dir=input("Please specify the directory that you store results.",'s');
    data_type=input("Please choose the data to read. Output:1; Returns:2; Strategy(single run):3 ; Probe_position:4 ; Plot probe_position:5 ; Exit:7.");
    cd(work_dir);
    switch data_type

        case 1 % Read output from training
            filename = 'output.csv';
            [results_output,name_output]=readcsv(filename);
            
            if results_output ==0
            else
                Episode = results_output(:,1);
                AvgDrag = -2*results_output(:,2);
                AvgLift = results_output(:,3);
                AvgRecArea = results_output(:,4);
                Num_Env = results_output(:,5);
                Epi_Reward = results_output(:,6);
                Num_Epi = Episode.*(1+Num_Env)+65*(Episode-1);
                [Num_Epi,I]=sort(Num_Epi);
                New_Reward = Epi_Reward(I);
                i1=1000;
            
                % Plot the figures
                % Plot AvgDrag vs Episode
                figure(1),plot(Num_Epi(i1:end),AvgDrag(i1:end),'LineWidth',2);
                title(['Average Drag per Episode']);
                ylabel([name_output{2},'(Cd)']);
                xlabel('Number of Episode');
                hold on
                % Plot AvgRecArea vs Episode
                figure(2),plot(Num_Epi(i1:end),AvgRecArea(i1:end),'LineWidth',2);
                title(['Average Recirculation Area per Episode']);
                ylabel([name_output{4}]);
                xlabel('Number of Episode');
                hold on 
                % Plot Episode Reward vs Episode

                figure(3),plot(Num_Epi(i1:end),Epi_Reward(i1:end),'LineWidth',2);
                %title(['Episode Reward vs Number of Episode']);
                ylabel('Episode Reward');
                xlabel('Number of Episode');
                legend('TQC-FM0FS-tri512','TQC-FM0FS-dou512','TQC-PM0FS-tri512')
                hold on 
            end
            
            continue
            
        case 2 % Read returns from training
            filename = 'returns_tf.csv';
            [results_returns,name_returns]=readcsv(filename);
            
            if results_returns ==0
                disp("No results are read.")
            else
                Episode = results_returns(:,1);
                Returns = results_returns(:,2);
                
                % Plot the figures
                % Plot Return vs Episode
                figure(1)
                plot(Episode,Returns,'LineWidth',1,'Color',Color_select(loop,:));
                title(['Return per Episode']);
                ylabel([name_returns{2}]);
                xlabel('Number of Episode');
                hold on 
                
                % Plot rolling average of Return per 50 episodes
                Avg_Returns = zeros(size(Episode,1),1);
                for i = 1:(size(Episode,1))
                    if i<=size(Episode,1)-49
                        Avg_Returns(i) = mean(Returns(i:i+49));
                    else
                        Avg_Returns(i) = mean(Returns(end-50:end));
                    end
                end
                p(loop)=plot(Episode,Avg_Returns,'LineWidth',1,'Color',Color_select(loop,1:3));
                %  legend([p(1),p(2),p(3),p(4),p(5)],{'FM-FNN','PM-FNN','PM-RNN','PM-RNN-SR','PM-RNN-CL'})
            end
            loop=loop+1;
            continue
            
        case 3 % Read control results from single test
            filename = 'test_strategy.csv';
            [results_strategy,name_strategy]=readcsv(filename);
            
            if results_strategy==0
                disp("No results are read.")
            else
                Time = 0.004*results_strategy(:,1);
                Cd = -2*results_strategy(:,2);
                %Cd = Cd(100001:200000);
                %Time = Time(100001:200000);
                Cl = results_strategy(:,3);
                Area = results_strategy(:,4);
                Jet_avg_v = results_strategy(:,5)/0.1;
                %Jet_avg_v = Jet_avg_v(100001:200000);
                Jet_Q = results_strategy(:,6);
                Cd_avg = ones(size(Cd,1),1)*mean(Cd(end-12500:end));
                % Plot the figures
                % Plot Cd
                figure(1)
                p(loop) = plot(Time,Cd,'LineWidth',1,'Color',Color_select(loop,1:3));
                %figure(1),plot(Time,Cl,'LineWidth',1,'Color',Color_select(loop,1:3));
                hold on
                figure(1),plot(Time,Cd_avg,'LineWidth',1.5,'LineStyle','--','Color',Color_select(loop,1:3));
                title(['Cd with final policy']);
                ylabel('Cd');
                xlabel('Non-dimensional time');
                %legend('Base-GRU','Base-Normal','Wake-GRU','Wake-Normal')
                %legend([p(1),p(2),p(3),p(4),p(5),p(6)],{'FM0FS-dou512','FM0FS-tri512','PM0FS-dou512','PM0FS-tri512','PM35FS-dou512','PM35FS-tri512'})
                hold on            
                % Plot Jets
                figure(2),plot(Time,Jet_avg_v,'LineWidth',1.5)
                %title(['Action in final policy (Mass flow of jets)']);
                ylabel('Average Jet Velocity');
                xlabel('Non-dimensional time');
                %legend('Base-GRU','Base-Normal','Wake-GRU','Wake-Normal')
                hold on
                % Obtain the fft of actions
                
                    Fs = 250; % The frequency we obtain states is 250Hz
                    s = Jet_avg_v(end-24999:end);
                    L = size(s,1); % The length of the signal
                    f = Fs*(0:(L/2))/L; % The frequency sampling rate
                    s_fft = fft(s);
                    P2 = abs(s_fft/L);
                    P1 = P2(1:L/2+1);
                    P1(2:end-1) = 2*P1(2:end-1);
                    figure(3);
                    plot(f(2:20),P1(2:20),'LineWidth',1.5)
                    hold on
                    title('Single-Sided Spectrum of Control Actuations')
                    xlabel('Non-dimensional Frequency')
                    ylabel('Amplitude')
            end
                % Plot base flow Cd
                %Cd_base = ones(size(Time,1),1) * 1.244;
                %figure(1), p(7) = plot(Time,Cd_base,'LineWidth',1.5,'LineStyle','-.')
                %  legend([p(1),p(2),p(3),p(4),p(5)],{'FM-NN','PM-FNN','PM-RNN','PM-RNN-SR','PM-RNN-CL'})
            loop = loop+1;
            continue

        case 4 % Read probe positions
            x_upstream = -4;
            x_downstream = 5.2;
            y_domain = 2.5;
            height_cylinder = 1;
            aspect_ratio = 1;

            length_cylinder = height_cylinder*aspect_ratio;
            filename = 'Probe.csv';
            [probe_positions,name_probe]=readcsv(filename);
            number=size(probe_positions,1);
            A = exist("positions","var");
            if A==0
                positions=probe_positions;
                num_probes=number;
            elseif A == 1
                positions=[positions,probe_positions];
                num_probes=[num_probes,number];
            end

             continue

         case 5 % Plot probe positions from training, do this after case 4
            B = exist("positions","var");
            if B==0
                disp("No results for plotting.")
            else
                iter=size(num_probes,2);
            end

            % Shape of domain
            pgon = polyshape([x_upstream,x_upstream,x_downstream,x_downstream],[y_domain,-y_domain,-y_domain,y_domain]);
            plot(pgon,'FaceColor',[1 1 1],'FaceAlpha',1,'EdgeColor','red');
            axis equal
            hold on

            % Shape of cylinder
            pgon2 = polyshape([-length_cylinder/2,-length_cylinder/2,length_cylinder/2,length_cylinder/2],[height_cylinder/2,-height_cylinder/2,-height_cylinder/2,height_cylinder/2]);
            plot(pgon2,'FaceColor',[1 1 1],'FaceAlpha',1,'EdgeColor','black');
            axis equal
            hold on

            for i = 1:iter
                x = positions(:,(3*i-1));
                y = positions(:,(3*i));
                p3=plot(x,y,'.');
                title('Sketch of flow environment');
                ylabel('Non-dimensional y');
                xlabel('Non-dimensional x');

                hold on
            end
            legend('Domain','Cylinder','Sensors')
            %legend('Domain','Cylinder','DS-2.0','DS-1.6','DS-1.2','DS-0.8','DS-0.4','Base')

            continue

        case 7
            stop=0; % Quit the loop
            
        otherwise
            disp('Please select the correct number.')
            continue
    end

end

%%%%%%%%%%
% The function used to read data from csv files
function [results,var_name] = readcsv(filename)

if isfile(filename)
    opts = detectImportOptions(filename);
    prev = preview(filename,opts);
    % Parameters for reading (check before reading)
    dim=size(prev);
    opts.SelectedVariableNames = [1:dim(2)];
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

