function [ Chains ] = MultiTargetTrack( detections, Observs, InitState )
%MULTITARGETTRACK Runs a fixed-lag tracking algorithm for multiple
% targets with missed observations and clutter

global Par;

% Initialise particle array and diagnostics (an array of particle arrays)
Chains = cell(Par.T, 1);
BestEsts = cell(Par.T, 1);
MoveTypes = cell(Par.T, 1);

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
    InitEst.origin = ones(Par.NumTgts, 1);
    InitEst.origin_time = ones(Par.NumTgts, 1);
    
end

% InitChain = Chain(1, InitEst.Copy);
InitChain = struct( 'particles', cell(1), 'posteriors', zeros(1) );
InitChain.particles{1} = InitEst.Copy;

% Loop through time
for t = 1:Par.T
    
    tic;
    
    disp('**************************************************************');
    disp(['*** Now processing frame ' num2str(t)]);
    
    if t==1
        [Chains{t}, BestEsts{t}, MoveTypes{t}] = MCMCFrame(t, t, {InitChain}, InitEst, Observs);
    else
%         [Chains{t}, BestEsts{t}, MoveTypes{t}] = MCMCFrame(t, min(t,Par.L), Chains(1:t), BestEsts{max(1,t-Par.S)}.Copy, Observs);
        [Chains{t}, BestEsts{t}, MoveTypes{t}] = MCMCFrame(t, min(t,Par.L), Chains(1:t), BestEsts{t-1}.Copy, Observs);
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



function [MC, BestEst, move_types] = MCMCFrame(t, L, PrevChains, PrevBest, Observs)
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

% MC = Chain(Par.NumIt, PrevBest);
MC = struct( 'particles', [], 'posteriors', [] );
MC.particles = cell(Par.NumIt, 1);
MC.posteriors = -inf(Par.NumIt, Par.NumTgts);
MC.particles{1} = PrevBest;

accept = zeros(4,1);
move_types = zeros(Par.NumIt,1);

posterior_store = -inf(Par.NumIt, Par.NumTgts);
reverse_kernel_store = -inf(Par.NumIt, Par.NumTgts);
origin_post_store = -inf(Par.NumIt, Par.NumTgts);
for j = 1:Par.NumTgts
    posterior_store(1, j) = SingTargPosterior(j, t, L, PrevBest, Observs);
    reverse_kernel_store(1, j) = PrevBest.Sample(j, t-1, L-1, Observs, true);
    origin_post_store(1, j) = SingTargPosterior(j, t-1, L-1, PrevBest, Observs);
end

% Loop through iterations
for ii = 2:Par.NumIt
    
    % Copy the previous estimates
    Old = MC.particles{ii-1}.Copy;
    New = Old.Copy;
    
    % Copy the probability stores
    MC.posteriors(ii,:) = MC.posteriors(ii-1,:);
    posterior_store(ii,:) = posterior_store(ii-1,:);
    reverse_kernel_store(ii,:) = reverse_kernel_store(ii-1,:);
    origin_post_store(ii,:) = origin_post_store(ii-1,:);
    
    % Restart chain if required
    if (mod(ii, Par.Restart)==1) && (ii > 1)
        
        k = max(1,t-1);
        weights = exp(sum(PrevChains{max(1,t-1)}.posteriors, 2));
        weights = weights / sum(weights);
        new_part = randsample(size(PrevChains{k}.particles, 1), 1, true, weights);
        Old = PrevChains{k}.particles{new_part}.Copy;
        Old.ProjectTracks(t);
        MC.particles{ii-1} = Old.Copy;
        New = Old.Copy;
        
        for j = 1:Par.NumTgts
            MC.posteriors(ii-1, j) = -inf;
            MC.posteriors(ii, j) = -inf;
            posterior_store(ii-1, j) = SingTargPosterior(j, t, L, New, Observs);
            reverse_kernel_store(ii-1, j) = New.Sample(j, t-1, L-1, Observs, true);
            origin_post_store(ii-1, j) = SingTargPosterior(j, t-1, L-1, New, Observs);
        end
        
    end
    
    % Randomly choose target
%     j = unidrnd(New.N);
    j = mod(ii, New.N)+1;
    
    % Randomly select move type
    if t > Par.L
        type_weights = [2 1 1 1];
    else
        type_weights = [1 0 0 0];
    end
    type = randsample(1:length(type_weights), 1, true, type_weights);
    move_types(ii) = type;
    
    % Switch on move type
    switch type
        
        case 1 % Single target, fixed-history
            
            % Choose proposal start-point
            d = unidrnd(L);
%             d = L;
            
            % Sample proposal
            new_ppsl = New.Sample(j, t, d, Observs, false);
            old_ppsl = Old.Sample(j, t, d, Observs, true);
            
            
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
            
%             % Generate origin states
%             so = t - Old.origin_time(j);
%             
%             OldOrigin = Old.Copy;
%             OldOrigin.tracks{j} = PrevChains{Old.origin_time(j)}.particles{Old.origin(j)}.tracks{j}.Copy;
            
            % Propose associations
            new_ppsl = New.Sample(j, t, L, Observs, false);
            old_ppsl = Old.Sample(j, t, L, Observs, true);
            
            
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

            % Work out do - the number of frames over which Old and OldOrigin differ
            for do = 1:L
                if OldOrigin.tracks{j}.Present(t-do)
                    if OldOrigin.tracks{j}.GetState(t-do) == Old.tracks{j}.GetState(t-do)
                        break
                    end
                end
            end

            % Sample proposal
            new_ppsl = (1/(L-sn+1)) * New.Sample(j, t, dn, Observs, false);
            old_ppsl = (1/(L-so+1)) * Old.Sample(j, t, do, Observs, true);
            
            
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
            
%             % Generate origin states
%             so = t - Old.origin_time(j);
%             OldOrigin = Old.Copy;
%             OldOrigin.tracks{j} = PrevChains{Old.origin_time(j)}.particles{Old.origin(j)}.tracks{j}.Copy;
            
            % Propose associations
            new_ppsl = New.SampleBridge(j, t, L, b, Observs, false);
            old_ppsl = Old.SampleBridge(j, t, L, b, Observs, true);

    end
                
    % Find origin posteriors and reverse kernels
    if ~(type==1)
        new_origin_post = SingTargPosterior(j, t-sn, L-sn, NewOrigin, Observs);
        if (sn < L)
            new_reverse_kernel = NewOrigin.Sample(j, t-sn, L-sn, Observs, true);
        else
            new_reverse_kernel = 0;
        end
    else
        new_origin_post = origin_post_store(ii-1,j);
        new_reverse_kernel = reverse_kernel_store(ii-1,j);
    end
    
    old_origin_post = origin_post_store(ii-1,j);
    old_reverse_kernel = reverse_kernel_store(ii-1,j);
    
    % Calculate posteriors
    new_post = SingTargPosterior(j, t, L, New, Observs);
%     old_post = SingTargPosterior(j, t, L, Old, Observs);
    old_post = posterior_store(ii-1, j);

    % Test for acceptance
    ap = (new_post - old_post) ...
        + (old_origin_post - new_origin_post) ...
        + (old_ppsl - new_ppsl) ...
        + (new_reverse_kernel - old_reverse_kernel);
    
    if log(rand) < ap
        MC.particles{ii} = New;
        MC.posteriors(ii,j) = new_post;
        accept(type) = accept(type) + 1;
        
        posterior_store(ii,j) = new_post;
        reverse_kernel_store(ii,j) = new_reverse_kernel;
        origin_post_store(ii,j) = new_origin_post;
        
    else
        MC.particles{ii} = Old;
        MC.posteriors(ii,j) = old_post;
        
        posterior_store(ii,j) = old_post;
%         ppsl_store(ii,j) = old_ppsl;
        reverse_kernel_store(ii,j) = old_reverse_kernel;
        origin_post_store(ii,j) = old_origin_post;
        
    end
    
end

% Pick the best particle
total_post = sum(MC.posteriors, 2);
best_ind = find(total_post==max(total_post), 1);
BestEst = MC.particles{best_ind}.Copy;
BestEst.origin(:) = best_ind;
BestEst.origin_time(:) = t;

disp(['*** Accepted ' num2str(accept(1)) ' fixed-history single target moves in this frame']);
disp(['*** Accepted ' num2str(accept(2)) ' full single target moves in this frame']);
disp(['*** Accepted ' num2str(accept(3)) ' full with preserved window-history single target moves in this frame']);
disp(['*** Accepted ' num2str(accept(4)) ' bridging-history single target moves in this frame']);

end

