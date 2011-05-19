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
    InitEst = TrackSet(1:size(InitState, 2), tracks);
    
else
    
    % Start with initial particle locations
    tracks = cell(size(InitState'));
    for j = 1:size(InitState, 2)
        tracks{j} = Track(0, 1, InitState(j), 0);
    end
    InitEst = TrackSet(1:size(InitState, 2), tracks);
    InitEst.origin = ones(Par.NumTgts);
    InitEst.origin_time = ones(Par.NumTgts);
    
end

InitChain = Chain(1, InitEst.Copy);

% Loop through time
for t = 1:Par.T
    
    tic;
    
    disp('**************************************************************');
    disp(['*** Now processing frame ' num2str(t)]);
    
    if t==1
        [Chains{t}, BestEsts{t}] = MCMCFrame(t, t, {InitChain}, InitEst, Observs);
    else
%         [Chains{t}, BestEsts{t}] = MCMCFrame(t, min(t,Par.L), Chains(1:t), BestEsts{max(1,t-Par.S)}.Copy, Observs);
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
b = 2;

PrevBest.ProjectTracks(t);
% for tt = t-s+1:t
%     PrevBest.ProjectTracks(tt);
% end

MC = Chain(Par.NumIt, PrevBest);

accept = zeros(4,1);

% Loop through iterations
for ii = 2:Par.NumIt
    
    Old = MC.particles{ii-1}.Copy;
    New = Old.Copy;
    
    % Randomly choose target
    j = unidrnd(New.N);
    
    % Randomly select move type
    if t > Par.L
        type_weights = [2 1 1 1];
    else
        type_weights = [1 0 0 0];
    end
    type = randsample(1:length(type_weights), 1, true, type_weights);
    
    % Switch on move type
    switch type
        
        case 1 % Single target, fixed-history
            
            % Choose proposal start-point
            d = unidrnd(L);
%             d = L;
            
            % Sample proposal
            new_ppsl = New.Sample(j, t, d, Observs, false);
            old_ppsl = Old.Sample(j, t, d, Observs, true);
            
            % Reverse Kernel
            new_reverse_kernel = 0;
            old_reverse_kernel = 0;
            
            old_origin_post = 0;
            new_origin_post = 0;
            
            
        case 2 % Single target, history and window - assumes independence for frames <= t-L
            
%             sn = s;
            sn = unidrnd(s);
            k = t-sn;
            
            % Propose a new origin and copy it
            new_part = unidrnd(size(PrevChains{k}.particles, 1));
            New.tracks{j} = PrevChains{k}.particles{new_part}.tracks{j}.Copy;
            NewOrigin = New.Copy;
            for tt = t-sn+1:t
                New.tracks{j}.Extend(tt, zeros(4,1), 0);
            end
            New.origin(j) = new_part;
            New.origin_time(j) = t-sn;
            
            % Generate origin states
            so = t - Old.origin_time(j);
            
            OldOrigin = Old.Copy;
            OldOrigin.tracks{j} = PrevChains{Old.origin_time(j)}.particles{Old.origin(j)}.tracks{j}.Copy;
            
%             % Propose associations
%             new_ppsl = New.Sample(j, t, sn, Observs, false);
%             old_ppsl = Old.Sample(j, t, so, Observs, true);
            
            % Propose associations
            new_ppsl = New.Sample(j, t, L, Observs, false);
            old_ppsl = Old.Sample(j, t, L, Observs, true);
            
            % Find origin posteriors
            new_origin_post = SingTargPosterior(j, t-sn, L-sn, NewOrigin, Observs);
            old_origin_post = SingTargPosterior(j, t-so, L-so, OldOrigin, Observs);
    
            % Reverse Kernel
%             new_reverse_kernel = 0;
%             old_reverse_kernel = 0;
            if (t > sn) && (sn < L)
                new_reverse_kernel = NewOrigin.Sample(j, t-sn, L-sn, Observs, true);
            else
                new_reverse_kernel = 0;
            end
            if (t > so) && (so < L)
                old_reverse_kernel = OldOrigin.Sample(j, t-so, L-so, Observs, true);
            else
                old_reverse_kernel = 0;
            end
            
            
        case 3 % Single target, history and window, keeping within-window history - assumes independence for frames <= t-L
            
%             sn = s;
            sn = unidrnd(s);
            dn = sn-1+unidrnd(L-sn+1);
            
            k = t-sn;
            
            % Propose a new origin and copy it
            new_part = unidrnd(size(PrevChains{k}.particles, 1));
            New.tracks{j} = PrevChains{k}.particles{new_part}.tracks{j}.Copy;
            NewOrigin = New.Copy;
            for tt = t-sn+1:t
                New.tracks{j}.Extend(tt, zeros(4,1), 0);
            end
            New.origin(j) = new_part;
            New.origin_time(j) = t-sn;
            
            % Generate origin states
            so = t - Old.origin_time(j);
            
            OldOrigin = Old.Copy;
            OldOrigin.tracks{j} = PrevChains{Old.origin_time(j)}.particles{Old.origin(j)}.tracks{j}.Copy;
            
%             % Propose associations
%             new_ppsl = New.Sample(j, t, sn, Observs, false);
%             old_ppsl = Old.Sample(j, t, so, Observs, true);

            % Work out do - the number of frames over which Old and OldOrigin differ
            for do = 1:L
                if OldOrigin.tracks{j}.Present(t-do)
                    if OldOrigin.tracks{j}.GetState(t-do) == Old.tracks{j}.GetState(t-do)
                        break
                    end
                end
            end

            % Propose associations
            new_ppsl = (1/(L-sn+1)) * New.Sample(j, t, dn, Observs, false);
            old_ppsl = (1/(L-so+1)) * Old.Sample(j, t, do, Observs, true);
            
            % Find origin posteriors
            new_origin_post = SingTargPosterior(j, t-sn, L-sn, NewOrigin, Observs);
            old_origin_post = SingTargPosterior(j, t-so, L-so, OldOrigin, Observs);
    
            % Reverse Kernel
%             new_reverse_kernel = 0;
%             old_reverse_kernel = 0;
            if (t > sn) && (sn < L)
                new_reverse_kernel = NewOrigin.Sample(j, t-sn, L-sn, Observs, true);
            else
                new_reverse_kernel = 0;
            end
            if (t > so) && (so < L)
                old_reverse_kernel = OldOrigin.Sample(j, t-so, L-so, Observs, true);
            else
                old_reverse_kernel = 0;
            end
            
            
        case 4 % Single target, history and bridging region - assumes independence for frames <= t-L
            
%             sn = s;
            sn = unidrnd(s);
            k = t-sn;
            
            % Propose a new origin and copy it
            new_part = unidrnd(size(PrevChains{k}.particles, 1));
            NewOrigin = New.Copy;
            New.tracks{j}.CopyHistory(t-L, PrevChains{k}.particles{new_part}.tracks{j});
            NewOrigin.tracks{j} = PrevChains{k}.particles{new_part}.tracks{j}.Copy;
            New.origin(j) = new_part;
            New.origin_time(j) = t-sn;
            
            % Generate origin states
            so = t - Old.origin_time(j);
            OldOrigin = Old.Copy;
            OldOrigin.tracks{j} = PrevChains{Old.origin_time(j)}.particles{Old.origin(j)}.tracks{j}.Copy;

            % Find origin posteriors
            new_origin_post = SingTargPosterior(j, t-sn, L-sn, NewOrigin, Observs);
            old_origin_post = SingTargPosterior(j, t-so, L-so, OldOrigin, Observs);
    
            % Reverse Kernel
%             new_reverse_kernel = 0;
%             old_reverse_kernel = 0;
            if (t > sn) && (sn < L)
                new_reverse_kernel = NewOrigin.Sample(j, t-sn, L-sn, Observs, true);
            else
                new_reverse_kernel = 0;
            end
            if (t > so) && (so < L)
                old_reverse_kernel = OldOrigin.Sample(j, t-so, L-so, Observs, true);
            else
                old_reverse_kernel = 0;
            end
            
            % Propose associations
            new_ppsl = New.SampleBridge(j, t, L, b, Observs, false);
            old_ppsl = Old.SampleBridge(j, t, L, b, Observs, true);

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
BestEst.origin(:) = best_ind;
BestEst.origin_time(:) = t;

disp(['*** Accepted ' num2str(accept(1)) ' fixed-history single target moves in this frame']);
disp(['*** Accepted ' num2str(accept(2)) ' full single target moves in this frame']);
disp(['*** Accepted ' num2str(accept(3)) ' full with preserved window-history single target moves in this frame']);
disp(['*** Accepted ' num2str(accept(4)) ' bridging-history single target moves in this frame']);

end

