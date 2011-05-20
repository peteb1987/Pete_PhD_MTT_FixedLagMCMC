% Base script for multi-frame multi-target tracker using fixed-lag MCMC

% Clear the workspace (maintaining breakpoints)
clup
dbstop if error

% Define all the necessary parameters in a global structure.
DefineParameters;

for rand_seed = 2:10
    
    % Set a standard random stream (for repeatability)
    s = RandStream('mt19937ar', 'seed', rand_seed);
    RandStream.setDefaultStream(s);
    
    % Specify target behaviour
    TargSpec = SpecifyTargetBehaviour;
    
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
    
    % % Analyse associations
    % [ass, count, present] = AnalyseAss( detections, Chains{Par.T}, Par.T);
    
end