classdef TrackSet < handle
    %TRACKSET A set of track objects, comprising a multi-target state over
    %the duration of a scene
    
    properties
        tracks          % Cell array of track objects
        N               % Number of tracks
        members         % Track ids
        origin          % Index of particle in previous chain from which this is estimate is derived
        origin_time     % The frame of the chain from which the estimate was originally proposed
        
    end
    
    methods
        
        % Constructor
        function obj = TrackSet(members, tracks)
            obj.tracks = tracks;
            obj.N = length(tracks);
            obj.members = members;
        end %Constructor
        
        % Copy
        function new = Copy(obj)
            t = cell(size(obj.tracks));
            for k = 1:length(t)
                t{k} = obj.tracks{k}.Copy;
            end
            new = TrackSet(obj.members, t);
            new.origin = obj.origin;
            new.origin_time = obj.origin_time;
        end
        
        
        
%         % Remove Track
%         function RemoveTrack(obj, j)
%             obj.tracks(j) = [];
%             obj.N = obj.N - 1;
%         end
%         
%         
%         
%         % Add Track
%         function AddTrack(obj, NewTrack)
%             obj.tracks = [obj.tracks; {NewTrack}];
%             obj.N = obj.N + 1;
%         end
        
        
        
        %ProjectTracks - Projects each track in a TrackSet forward by one frame.
        function ProjectTracks(obj, t)
            
            global Par;
            
            % Loop through targets
            for j = 1:obj.N
                
                if obj.tracks{j}.death == t
                    % Only extend it if it dies in the current frame. If
                    % not, its expired completely.
                    
                    % % Check that the track ends at t-1
                    % assert(obj.tracks{j}.death==t, 'Track to be projected does not end at t-1');
                    
                    % Get previous state
                    prev_state = obj.tracks{j}.GetState(t-1);
                    
                    % Project it forward
                    state = Par.A * prev_state;
                    
                    % Extend the track
                    obj.tracks{j}.Extend(t, state, 0);
                    
                end
                
            end
            
        end
        
        
        
        % SampleStates - Propose changes to state and calculate
        % proposal probability
        function ppsl_prob = SampleStates(obj, j, t, L, Observs, just_prob)
            
            % just_prob is a flag indicating that we should just calculate
            % the probability of the current values, not sample new ones
            
            global Par;
            
            NewTracks = cell(obj.N, 1);
            
            % Only need examine those which are present after t-L
            if obj.tracks{j}.death > t-L+1
                
                % How long should the KF run for?
                last = min(t, obj.tracks{j}.death - 1);
                first = max(t-L+1, obj.tracks{j}.birth+1);
                num = last - first + 1;
                
                % Draw up a list of associated hypotheses
                obs = ListAssocObservs(last, num, obj.tracks{j}, Observs);
                
                % Run a Kalman filter the target
                [KFMean, KFVar] = KalmanFilter(obs, obj.tracks{j}.GetState(first-1), Par.KFInitVar*eye(4));
                
                % Sample Kalman filter
                if ~just_prob
                    [NewTracks{j}, ppsl_prob] = SampleKalman(KFMean, KFVar);
                else
                    track = obj.tracks{j}.Copy;
                    [NewTracks{j}, ppsl_prob] = SampleKalman(KFMean, KFVar, track);
                end
                
                if ~just_prob
                    % Update distribution
                    obj.tracks{j}.Update(last, NewTracks{j}, []);
                end
                
            end
            
        end
        
        
        
        %SampleAssociations - Propose changes to associations and calculate
        % proposal probability
        function ppsl_prob = SampleAssociations(obj, j, t, L, Observs, just_prob)

            % just_prob is a flag indicating that we should just calculate
            % the probability of the current values, not sample new ones
            
            ppsl_prob = 0;
            
            jah_ppsl = SampleJAH(j, t, L, obj, Observs, just_prob);
            ppsl_prob = ppsl_prob + sum(jah_ppsl(:));
            
        end
        
        
        
        %SampleSearchAssociations - As above, but for the search track
        function ppsl_prob = SampleSearchAssociations(obj, t, L, Observs, BirthSites )
            
            jah_ppsl = SampleSearchJAH(t, L, obj, Observs, BirthSites);
            ppsl_prob = sum(jah_ppsl(:));
            
        end
        
    end
    
end

