function [ Chains ] = MultiTargetTrack( detections, Observs, InitState )
%MULTITARGETTRACK Runs a fixed-lag tracking algorithm for multiple
% targets with missed observations and clutter

global Par;

% Initialise particle array and diagnostics (an array of particle arrays)
Chains = cell(Par.T, 1);
BestEsts = cell(Par.T, 1);

if ~Par.FLAG_InitTargs
    
    % No knowledge of target starting positions
    tracks = cell(0,1);
    
else
    
    % Start with initial particle locations
    tracks = cell(size(InitState'));
    for j = 1:size(InitState, 2)
        tracks{j} = Track(0, 1, InitState(j), 0);
    end
    
end

InitEst = TrackSet(1:size(InitState, 2), tracks);
InitEst.origin = 1;
InitEst.origin_time = 1;
InitChain = Chain(1, InitEst.Copy);

% Loop through time
for t = 1:Par.T
    
    tic;
    
    disp('**************************************************************');
    disp(['*** Now processing frame ' num2str(t)]);
    
    if t==1
        [Chains{t}, BestEsts{t}] = MCMCFrame(t, t, {InitChain}, InitEst, Observs);
    else
%         [Chains{t}, BestEsts{t}] = MCMCFrame(t, min(t,Par.L), Chains(1:t), BestEsts{max(1,t-Par.S)}, Observs);
        [Chains{t}, BestEsts{t}] = MCMCFrame(t, min(t,Par.L), Chains(1:t), BestEsts{t-1}.Copy, Observs);
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



function [MC, BestEst] = MCMCFrame(t, L, PrevChains, PrevBest, Observs)
% Execute a frame of the fixed-lag MCMC target tracker

% t - latest time frame
% L - window size
% PrevChains - MCs from the t-L to t-1 processing step, used to propose history changes
% PrevBest - The max-posterior estimate from the previous chain - used for state history <=t-L
% Observs - observations

global Par;

s = min(t,Par.S);

PrevBest.ProjectTracks(t);
% for tt = t-s+1:t
%     PrevBest.ProjectTracks(tt);
% end

MC = Chain(Par.NumIt, PrevBest);

accept = zeros(2,1);

% Loop through iterations
for ii = 2:Par.NumIt
    
    Old = MC.particles{ii-1}.Copy;
    New = Old.Copy;
    
    % Randomly choose target
    j = unidrnd(New.N);
    
    % Randomly select move type
    if t > Par.L
        type_weights = [1 1];
    else
        type_weights = [1 0];
    end
    type = randsample(1:2, 1, true, type_weights);
    
    % Switch on move type
    switch type
        
        case 1 % Single target, fixed-history
            
            % Choose proposal start-point
            d = unidrnd(L);
%             d = L;
%             if t<Par.L
%                 d = L;
%             end
            
            % Propose associations
            new_assoc_ppsl = New.SampleAssociations(j, t, d, Observs, false);
            old_assoc_ppsl = Old.SampleAssociations(j, t, d, Observs, true);
            
            % Propose states
            new_state_ppsl = New.SampleStates(j, t, d, Observs, false);
            old_state_ppsl = Old.SampleStates(j, t, d, Observs, true);
            
            old_ppsl = old_state_ppsl + old_assoc_ppsl;
            new_ppsl = new_state_ppsl + new_assoc_ppsl;
            
            % Reverse Kernel
            new_reverse_kernel = 0;
            old_reverse_kernel = 0;
            
            old_origin_post = 0;
            new_origin_post = 0;
            
        case 2 % Single target, history and window - assumes independence for frames <= t-L
            
            sn = s;
%             sn = unidrnd(s);
            k = t-sn;
            
            % Propose a new origin and copy it
            new_part = unidrnd(size(PrevChains{k}.particles, 1));
            New.tracks{j} = PrevChains{k}.particles{new_part}.tracks{j}.Copy;
            NewOrigin = New.Copy;
            for tt = t-sn+1:t
                New.tracks{j}.Extend(tt, zeros(4,1), 0);
            end
            New.origin = new_part;
            New.origin_time = t-sn;
            
            % Generate origin states
            so = t - Old.origin_time;
            
            OldOrigin = Old.Copy;
            OldOrigin.tracks{j} = PrevChains{Old.origin_time}.particles{Old.origin}.tracks{j}.Copy;
            
            % Propose associations
            new_assoc_ppsl = New.SampleAssociations(j, t, sn, Observs, false);
            old_assoc_ppsl = Old.SampleAssociations(j, t, so, Observs, true);
            
            % Propose states
            new_state_ppsl = New.SampleStates(j, t, sn, Observs, false);
            old_state_ppsl = Old.SampleStates(j, t, so, Observs, true);
            
%             % Propose associations
%             new_assoc_ppsl = New.SampleAssociations(j, t, L, Observs, false);
%             old_assoc_ppsl = Old.SampleAssociations(j, t, L, Observs, true);
%             
%             % Propose states
%             new_state_ppsl = New.SampleStates(j, t, L, Observs, false);
%             old_state_ppsl = Old.SampleStates(j, t, L, Observs, true);
            
            old_ppsl = old_state_ppsl + old_assoc_ppsl;
            new_ppsl = new_state_ppsl + new_assoc_ppsl;
            
            % Find origin posteriors
            new_origin_post = SingTargPosterior(j, t-sn, L-sn, NewOrigin, Observs);
            old_origin_post = SingTargPosterior(j, t-so, L-so, OldOrigin, Observs);
%             new_origin_post = SingTargPosterior(j, t-sn, L-sn, NewOrigin, Observs);
%             old_origin_post = SingTargPosterior(j, t-so, L-so, OldOrigin, Observs);

            if ((t==6)&&(ii==155)) || ...
                    ((t==7)&&(ii==159)) || ...
                    ((t==8)&&(ii==254)) || ...
                    ((t==9)&&(ii==16)) || ...
                    ((t==10)&&(ii==2)) || ...
                    ((t==11)&&(ii==11)) || ...
                    ((t==12)&&(ii==42))
                old_origin_post = old_origin_post;20000;
            end
    
            % Reverse Kernel
            new_reverse_kernel = 0;
            old_reverse_kernel = 0;
%             if (t > sn) && (sn < L)
%                 new_reverse_kernel = NewOrigin.SampleAssociations(j, t-sn, L-sn, Observs, true);
%                 new_reverse_kernel = new_reverse_kernel + NewOrigin.SampleStates(j, t-sn, L-sn, Observs, true);
%             else
%                 new_reverse_kernel = 0;
%             end
%             if (t > so) && (so < L)
%                 old_reverse_kernel = OldOrigin.SampleAssociations(j, t-so, L-so, Observs, true);
%                 old_reverse_kernel = old_reverse_kernel + OldOrigin.SampleStates(j, t-so, L-so, Observs, true);
%             else
%                 old_reverse_kernel = 0;
%             end
            
    end
    
    % Calculate posteriors
    new_post = SingTargPosterior(j, t, L, New, Observs);
    old_post = SingTargPosterior(j, t, L, Old, Observs);

    % Test for acceptance
    ap = (new_post - old_post) ...
        + (old_origin_post - new_origin_post) ...
        + (old_ppsl - new_ppsl) ...
        + (new_reverse_kernel - old_reverse_kernel);
    
    if log(rand) < ap
        MC.particles{ii} = New;
        MC.posteriors(ii) = new_post;
        accept(type) = accept(type) + 1;
    else
        MC.particles{ii} = Old;
        MC.posteriors(ii) = old_post;
    end
    
end

% Pick the best particle
best_ind = find(MC.posteriors==max(MC.posteriors), 1);
BestEst = MC.particles{best_ind}.Copy;
BestEst.origin = best_ind;
BestEst.origin_time = t;

disp(['*** Accepted ' num2str(accept(1)) ' fixed-history single target moves in this frame']);
disp(['*** Accepted ' num2str(accept(2)) ' full single target moves in this frame']);

end

