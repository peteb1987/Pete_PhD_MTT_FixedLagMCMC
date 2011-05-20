classdef Chain
    %CHAIN Contains elements of a Markov Chain in multi-target space
    
    properties
        particles
        posteriors
        orders           % Order in which target proposals occured
        origins          % Index of the particle in the previous chain from which the estimate was derived
        
    end
    
    methods
        
        % Constructor
        function obj = Chain(chain_len, init)
            blank = TrackSet([], {});
            obj.particles = cell(chain_len, 1);
            for ii=1:chain_len
                obj.particles{ii} = blank;
            end
            obj.particles{1} = init;
            obj.posteriors = -inf*ones(chain_len, init.N);
            obj.orders = zeros(chain_len, 1);
            obj.origins = zeros(chain_len, 1);
        end %Constructor
        
        
    end
    
end

