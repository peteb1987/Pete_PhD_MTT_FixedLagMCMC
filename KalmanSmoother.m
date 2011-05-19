function [ FinalState, FinalVar ] = KalmanSmoother( obs, init_state, init_var, final_state )
%KALMANFILTER Kalman filter a set of observations to give a track estimate,
%given the first and last states in the sequence.

%[ BackState, BackVar ]

global Par;

if size(obs, 1) == 0
    FinalState = cell(0,1);
    FinalVar = cell(0,1);
    return
end

L = length(obs(:,1)) + 2;

PredState = cell(L, 1);
PredVar = cell(L, 1);

EstState = cell(L, 1);
EstVar = cell(L, 1);

% Set C depending on model
if Par.FLAG_ObsMod == 0
    C = Par.C;

elseif Par.FLAG_ObsMod == 1
    % Use EKF approximation
    C = zeros(2, 4);
    
end

EstState{1} = init_state;
EstVar{1} = init_var;

% Loop through time
for k = 2:L-1
    
    % Prediction step
    
    PredState{k} = Par.A * EstState{k-1};
    PredVar{k} = Par.A * EstVar{k-1} * Par.A' + Par.Q;
    
    % Update step
    
    if length(obs{k-1})==2
        
        % Observation associated with target
        
        if (Par.FLAG_ObsMod == 1)
            
            % Linearisation
            x1 = PredState{k}(1);
            x2 = PredState{k}(2);
            C(1,1) = -x2/(x1^2+x2^2);
            C(1,2) = x1/(x1^2+x2^2);
            C(2,1) = x1/sqrt(x1^2+x2^2);
            C(2,2) = x2/sqrt(x1^2+x2^2);
            
        end
        
        % Innovation
        if Par.FLAG_ObsMod == 0
            y = obs{k-1} - C * PredState{k};
        elseif Par.FLAG_ObsMod == 1
            [bng, rng] = cart2pol(PredState{k}(1), PredState{k}(2));
            y = obs{k-1} - [bng; rng];
            if y(1) > pi
                y(1) = y(1) - 2*pi;
            elseif y(1) < -pi
                y(1) = y(1) + 2*pi;
            end
        end
        s = C * PredVar{k} * C' + Par.R;
        gain = PredVar{k} * C' / s;
        EstState{k} = PredState{k} + gain * y;
        EstVar{k} = (eye(4)-gain*C) * PredVar{k};
        
    elseif isempty(obs{k-1})
        
        % No observation
        
        EstState{k} = PredState{k};
        EstVar{k} = PredVar{k};
        
    else
        
        error('Observation model allows for 0 or 1 observations only');
        
    end

end

BackState = cell(L, 1); BackVar = cell(L, 1);
BackState{L} = final_state; BackVar{L} = zeros(4);
BackPredState = cell(L, 1); BackPredVar = cell(L, 1);
invA = inv(Par.A);
invQ = (Par.A\Par.Q)/(Par.A');
% Backwards
for k = L-1:-1:2
    
    % Prediction step
    BackPredState{k} = invA * BackState{k+1};
    BackPredVar{k} = invA * BackVar{k+1} * invA' + invQ;
    
    % Update step
    if length(obs{k-1})==2
        % Observation associated with target
        if (Par.FLAG_ObsMod == 1)
            % Linearisation
            x1 = BackPredState{k}(1);
            x2 = BackPredState{k}(2);
            C(1,1) = -x2/(x1^2+x2^2);
            C(1,2) = x1/(x1^2+x2^2);
            C(2,1) = x1/sqrt(x1^2+x2^2);
            C(2,2) = x2/sqrt(x1^2+x2^2);
        end
        
        % Innovation
        if Par.FLAG_ObsMod == 0
            y = obs{k-1} - C * BackPredState{k};
        elseif Par.FLAG_ObsMod == 1
            [bng, rng] = cart2pol(BackPredState{k}(1), BackPredState{k}(2));
            y = obs{k-1} - [bng; rng];
            if y(1) > pi
                y(1) = y(1) - 2*pi;
            elseif y(1) < -pi
                y(1) = y(1) + 2*pi;
            end
        end
        s = C * BackPredVar{k} * C' + Par.R;
        gain = BackPredVar{k} * C' / s;
        BackState{k} = BackPredState{k} + gain * y;
        BackVar{k} = (eye(4)-gain*C) * BackPredVar{k};
        
    elseif isempty(obs{k-1})
        % No observation
        
        BackState{k} = BackPredState{k};
        BackVar{k} = BackPredVar{k};
    else
        error('Observation model allows for 0 or 1 observations only');
    end
end


FinalState = cell(L,1);
FinalVar = cell(L,1);
for k = 2:L-1
    
    FinalVar{k} = inv( inv(EstVar{k}) + inv(BackVar{k}) );
    FinalState{k} =  FinalVar{k}*( EstVar{k}\EstState{k} + BackVar{k}\BackState{k} );

end

FinalState(1) = []; FinalState(end) = [];
FinalVar(1) = []; FinalVar(end) = [];

end

