addpath CostFunctions
addpath MEX
addpath Models
addpath Observers
addpath Storage
addpath Utils
addpath Wrappers


%% Script configuration
% folder name where the data for current experiment will be saved
settings.ExperimentName = 'BZ_dim_32_N_25_norm';

% true - show the figures, false - do not show the figures
settings.ShowFigures = true;

% true - display information about intermediate progress,
% false - do not show any messages
settings.ShowMessages = true;

% true - use bootstrapping, false - do not use bootstrapping
settings.UseBootstrap = true; 

% CIL settings
N = 25; % number of patterns in one subset
M = 18; % the dimension of eCDF vector
Ncurves = 1000; % number of eCDF vectors to compute mean and covariance
                     
% model    
model = BZModel(32);
% comment the next line if tne normalization is not used
model.Observer = MinMaxObserver(); % min-max normalization of data
                    
%% Initialization

% initialize storage manager for 
storage = StorageManager(settings.ExperimentName, 'CIL_simu');

% determine how many simulations we need to get the desired number of
% eCDF pairs
nsim = nlEstimateSimulationNumber(Ncurves);
   
% Next we estimate the constants (R0 and base), which are needed for the 
% construction of CIL likelihood. 

% Creating two sets of patterns
tic;
S1 = model.simulate(N,[]);
S2 = model.simulate(N,[]);
toc;

% Computing the distances between S1 and S2
tic;
dist = model.distance(S1,S2);
toc;

% Estimating the constants
[R0,base] = nlEstimateECDFConstants(dist,M,false);

% If specified in settings we create and display an eCDF vector, determined 
% by the sets S1 and S2.
if (settings.ShowFigures)       
    % creating eCDF vector
    [c0,x0] = nlCreateECDF(dist,M,false,R0,base);
    
    % displaying the result
    f1=figure(1);
    clf      
    hold on
    plot(x0,c0,'o-','color','r');
    xlabel('k')
    title('ECDF vectors')
    
    pause
end

%% Generation of pattern data
if settings.ShowMessages
    disp('Beginning pattern data generation...')
end

% creating the 3D array for storing measurements
if (~settings.UseBootstrap)
    ObsArray = zeros(size(S,1),N,nsim);
    ObsArray(:,:,1)=S1;
    ObsArray(:,:,2)=S2;

    % filling the array with measurements
    for i=3:nsim   
        ObsArray(:,:,i) = model.simulate(N,[]);
       
        if settings.ShowMessages
            fprintf('Simulation %i / %i has been finished. \n',i,nsim);
        end
    end
else
    ObsArray = zeros(size(S1,1),N,2);
    ObsArray(:,:,1)=S1;
    ObsArray(:,:,2)=S2;   
end

if settings.ShowMessages
    disp('Finishing pattern data generation...')
end

save(storage.createLocalPath('ObsArray.mat'),'ObsArray');  
    
%% CIL training set generation

% creating array for storing eCDF vectors.
Y = zeros(Ncurves,M+1);
k=1; %counter variable


% performance fix in the case of bootstrapping (computing the distances 
% beforehand)
if (settings.UseBootstrap)
    Sgen = [ObsArray(:,:,1) ObsArray(:,:,2)];
    dist_bank = model.distance(single(Sgen),single(Sgen));
end

if settings.ShowMessages
    disp('Beginning training set generation...')
end

% training set generation
for i=1:nsim
   
    if(k>Ncurves)
        break;
    end
    
    for j=i+1:nsim
        if(k>Ncurves)
            break;
        end
        
        % extracting the subset pair from ObsArray.
        if (~settings.UseBootstrap)
            S0 = ObsArray(:,:,i);
            S1 = ObsArray(:,:,j);
            % Computing distance matrix of the data.
            dist = model.distance(single(S0),single(S1));
        else           
            Sind = randperm(2*N);
            Sind1 = Sind(1:N);
            Sind2 = Sind(N+1:end);
            
            Sind3 = randi(N,1,N);
            Sind4 = randi(N,1,N);
                                           
            dist = dist_bank(Sind1(Sind3),Sind2(Sind4));                               
        end
        
        % from distance matrix we create empcdf curve
        [c0,x0] = nlCreateECDF(dist,M,false,R0,base);

        % plotting the results (if specified in script settings)
        if (settings.ShowFigures)
            plot(x0,c0,'o-','color','b');
            drawnow;
        end

        % saving values of eCDF vector
        Y(k,:)=c0;
                   
        if (settings.ShowMessages)
            fprintf('k = %i / %i \n', k, Ncurves);
        end
        
        k=k+1; 
    end
end

if settings.ShowMessages
    disp('Training set generation has been completed.')
end

if (settings.ShowFigures)
    hold off
end

%% Chi-squared Normality test

save(storage.createLocalPath('Y.mat'),'Y')

% removing tail values to aviod singular covariance matrix
[rngindex1,rngindex2] = nlGetThreshold(Y,1e-2,1e-2);
Y=Y(:,rngindex1:rngindex2);

% computing mean and covariance
Yave = mean(Y);
C=cov(Y);
iC = inv(C);

% chi-squared testing
Y0 = Y-repmat(Yave,size(Y,1),1);

khi2= sum(Y0'.*(iC*Y0')); 
[khi,x]=hist(khi2,25);
khi_n = khi/sum(khi)/(x(2)-x(1));

f2=figure(2);
clf
hold on
bar(x,khi_n)
plot(x,chi2pdf(x,size(Y,2)),'r','linewidth',3)
title('Chi-squared Normality test')
hold off

%% Data saving

% saving figures
print(f1, storage.createLocalPath('Fig1'), '-djpeg');
print(f2, storage.createLocalPath('Fig2'), '-djpeg');

% saving the CIL data for future use by other scripts
S0 = ObsArray(:,:,1);

curve.R0 = R0;
curve.base = base;
curve.nr = M;

save(storage.createLocalPath('CIL_simu.mat'),...
    'model',...
    'S0',...
    'Yave',...
    'iC',...
    'curve',...
    'N',...
    'rngindex1',...
    'rngindex2')

if settings.ShowMessages
    disp('All done!')
end





