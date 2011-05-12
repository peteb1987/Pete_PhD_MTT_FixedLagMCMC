% Base script for multi-frame multi-target tracker using fixed-lag MCMC

% Clear the workspace (maintaining breakpoints)
clup
dbstop if error

% Define all the necessary parameters in a global structure.
DefineParameters;

% Set a standard random stream (for repeatability)
s = RandStream('mt19937ar', 'seed', Par.rand_seed);
RandStream.setDefaultStream(s);

% Specify target behaviour
TargSpec = SpecifyTargetBehaviour;

for i=1:4
    [~]=unidrnd(50);
    [~] = unifrnd(0.15*Par.Xmax, 0.25*Par.Xmax);
    [~] = unifrnd(-pi, pi);
    [~] = unifrnd(-pi, pi);
    [~] = unifrnd(-pi, pi);
end

% Generate target motion
[TrueState, TargSpec] = GenerateTargetMotion(TargSpec);

% Generate observations from target states
[Observs, detections] = GenerateObs(TrueState);

% Plot states and observations
fig = PlotTrueState(TrueState);
PlotObs(Observs, detections);

% Run tracker
[ Chains ] = MultiTargetTrack(detections, Observs, {TargSpec(:).state} );

% Plot final estimates
PlotTracks(Chains{50}, fig);

% Analyse associations
[ass, count, present] = AnalyseAss( detections, Chains{Par.T}, Par.T);