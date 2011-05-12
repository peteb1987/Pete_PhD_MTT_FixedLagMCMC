function [ Chains ] = MultiTargetTrack( detections, Observs, InitState )
%MULTITARGETTRACK Runs a fixed-lag tracking algorithm for multiple
% targets with missed observations and clutter

global Par;

% Initialise particle array and diagnostics (an array of particle arrays)
Chains = cell(Par.T, 1);
BestEsts = cell(Par.T, 1);

if ~Par.FLAG_InitTargs
    
    % No knowledge of target starting positions
    InitEst = TrackSet([], cell(0,1));
    
else
    
    % Start with initial particle locations
    tracks = cell(size(InitState'));
    for j = 1:size(InitState, 2)
        tracks{j} = Track(0, 1, InitState(j), 0);
    end
    
end

InitEst = TrackSet(1:size(InitState, 2), tracks);
InitEst.origin = 1;
InitChain = Chain(1, InitEst.Copy);

% Loop through time
for t = 1:Par.T
    
    tic;
    
    disp('**************************************************************');
    disp(['*** Now processing frame ' num2str(t)]);
    
    if t==1
        [Chains{t}, BestEsts{t}] = MCMCFrame(t, t, InitChain, InitEst, Observs);
    else
        [Chains{t}, BestEsts{t}] = MCMCFrame(t, min(t,Par.L), Chains{t-1}, BestEsts{t-1}, Observs);
    end
    
    disp(['*** Correct associations at frame ' num2str(t-min(t,Par.L)+1) ': ' num2str(detections(t-min(t,Par.L)+1,:))]);
    assoc = [];
    for j = 1:Par.NumTgts
        get_ass = cellfun(@(x) x.tracks{j}.GetAssoc(t-min(t,Par.L)+1), Chains{t}.particles);
        mode_ass = mode(get_ass);
        assoc = [assoc, mode_ass];
    end
    disp(['*** Modal associations at frame ' num2str(t-min(t,Par.L)+1) ': ' num2str(assoc)]);
    assoc = cellfun(@(x) x.GetAssoc(t-min(t,Par.L)+1), BestEsts{t}.tracks)';
    disp(['*** MAP associations at frame ' num2str(t-min(t,Par.L)+1) ': ' num2str(assoc)]);
    
    disp(['*** Frame ' num2str(t) ' processed in ' num2str(toc) ' seconds']);
    disp('**************************************************************');
    
%     if mod(t, 100)==0
%         PlotTracks(Distns{t});
% %         plot(Observs(t).r(:, 2).*cos(Observs(t).r(:, 1)), Observs(t).r(:, 2).*sin(Observs(t).r(:, 1)), 'x', 'color', [1,0.75,0.75]);
% %         saveas(gcf, ['Tracks' num2str(t) '.eps'], 'epsc2');
% %         close(gcf)
%         pause(1);
%     end
    
end




end



function [MC, BestEst] = MCMCFrame(t, L, PrevChain, PrevBest, Observs)
% Execute a frame of the fixed-lag MCMC target tracker

% t - latest time frame
% L - window size
% PrevChain - MC from the t-1 processing step, used to propose large changes
% PrevBest - The max-posterior estimate from the previous chain - used for state history <=t-L
% Observs - observations

global Par;

PrevBest.ProjectTracks(t);
MC = Chain(Par.NumIt, PrevBest);

accept = zeros(2,1);
bad_origin_moves = 0;

% Loop through iterations
for ii = 2:Par.NumIt
    
    Old = MC.particles{ii-1}.Copy;
    New = Old.Copy;
    
    % Randomly choose target and proposal length
    j = unidrnd(New.N);
    d = unidrnd(L);
%     d = L;
    
    % Randomly select move type
    type = randsample(1:2, 1, true, [1 0]);
    
    % Switch on move type
    switch type
        
        case 1 % Single target, new proposal
            
            % Propose new associations
            new_assoc_ppsl = New.SampleAssociations(j, t, d, Observs, false);
            old_assoc_ppsl = Old.SampleAssociations(j, t, d, Observs, true);
            
            % Propose new states
            new_state_ppsl = New.SampleStates(j, t, d, Observs, false);
            old_state_ppsl = Old.SampleStates(j, t, d, Observs, true);
            
        case 2 % Single target, swap to t-1 path - THIS ISN'T VALID. IT CHANGES t-L state!
            
            new_part = unidrnd(size(PrevChain.particles, 1));
            New.tracks{j} = PrevChain.particles{new_part}.tracks{j}.Copy;
            New.tracks{j}.Extend(t, Par.A*New.tracks{j}.GetState(t-1), 0)
            New.origin = new_part;
            
            OriginEst = Old.Copy;
            OriginEst.tracks{j} = PrevChain.particles{Old.origin}.tracks{j}.Copy;
            OriginEst.tracks{j}.Extend(t, Par.A*OriginEst.tracks{j}.GetState(t-1), 0)
            new_assoc_ppsl = OriginEst.SampleAssociations(j, t, d, Observs, true);
            new_state_ppsl = OriginEst.SampleStates(j, t, d, Observs, true);

            old_assoc_ppsl = Old.SampleAssociations(j, t, d, Observs, true);
            old_state_ppsl = Old.SampleStates(j, t, d, Observs, true);
            
            % Prevent move if we change t-L state
            if New.tracks{j}.GetState(t-L)~=Old.tracks{j}.GetState(t-L)
                new_state_ppsl = inf;
                bad_origin_moves = bad_origin_moves + 1;
            end
    
    end
    
    % Calculate new posterior
    new_post_prob = SingTargPosterior(j, t, d, New, Observs);
    old_post_prob = SingTargPosterior(j, t, d, Old, Observs);

    % Test for acceptance
    ap = (new_post_prob - old_post_prob) + ...
        ((old_state_ppsl + old_assoc_ppsl) - (new_state_ppsl + new_assoc_ppsl));
    
    if log(rand) < ap
        MC.particles{ii} = New;
        MC.posteriors(ii) = new_post_prob;
        accept(type) = accept(type) + 1;
    else
        MC.particles{ii} = Old;
        MC.posteriors(ii) = old_post_prob;
    end
    
    
end

% Pick the best particle
best_ind = find(MC.posteriors==max(MC.posteriors), 1);
BestEst = MC.particles{best_ind}.Copy;
BestEst.origin = best_ind;

disp(['*** Accepted ' num2str(accept(1)) ' simple single target moves in this frame']);
disp(['*** Accepted ' num2str(accept(2)) ' origin-jump single target moves in this frame']);
disp(['*** ' num2str(bad_origin_moves) ' bad origin-jump moves']);

end

