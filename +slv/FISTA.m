function [x_star,cost,L] = FISTA(x0, A, b, varargin)

    % set defaults
    defaults.niter = 5; % number of iterations
    defaults.L = []; % Lipschitz constant
    defaults.R = []; % regularizers
    defaults.update_fun = []; % iteration udpate fun
    
    % parse inputs
    args = vararg_pair(defaults,varargin);

    if isempty(args.L)
        % run power iteration with defaults
        args.L = utl.L_pwritr(A);
    end
    L = args.L; % to return Lipschitz constant

    % initialize cost
    cost = zeros(1,args.niter+1);

    % define the cost function
    function c = cost_fun(x)
        % calculate data fidelity term
        c = 0.5 * norm(A * x - b, 2)^2;

        % add regularization norms
        if ~isempty(args.R) && iscell(args.R)
            for i = 1:length(R)
                c = c + args.R{i}.norm(x);
            end
        elseif ~isempty(args.R)
            c = c + args.R.norm(x);
        end
    end

    % initialize variables
    x_star = x0;
    t = 1;
    y = x0;
    cost(1) = cost_fun(x_star);

    for n = 1:args.niter
        time_itr = tic; % start timer

        % store old values
        x_old = x_star;
        t_old = t;

        % calculate the gradient and step
        g = reshape(A' * (A*y - b), size(x0));
        x_star = y - g/args.L;

        % apply the regularization
        if ~isempty(args.R) && iscell(args.R) % for multiple regularizers
            for j = 1:length(args.R)
                x_star = args.R{j}.prox(x_star);
            end
        elseif ~isempty(args.R) % for a single regularizer
            x_star = args.R.prox(x_star);
        end

        % calculate new step size
        t = (1 + sqrt(1+4*t_old^2))/2;

        % update y
        y = x_star + (t_old-1)/t * (x_star - x_old);

        % calculate the cost function
        time_cost = tic; % time the cost calculation
        cost(n+1) = cost_fun(x_star);
        time_itr = time_itr - toc(time_cost); % remove cost from total time
        
        % print update
        if ~isempty(args.update_fun)
            time_itr = toc(time_itr); % stop the clock
            args.update_fun(n,cost,x_star,time_itr);
        end

        % set a variable called "exititr" to exit at current iteration
        % when debugging
        if exist('exititr','var')
            break 
        end
        
    end

end