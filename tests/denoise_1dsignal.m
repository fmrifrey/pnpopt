N = 5000;

% design the signal to have specific features:
x_true = zeros(N,1);
x_true = x_true + 0.8*sin(2*pi*3.7*linspace(0,1,N)'); % constant low freq component
% x_true = x_true + 0.3*sin(2*pi*21*linspace(0,1,N)'); % constant high freq component
x_true = x_true + 2*tanh(10*pi*linspace(-1,1,N)'); % edge at N/2 (good for testing edge preservation)
x_true = x_true + 0.8*sin(2*pi*3.7*cos(2*pi*0.4*linspace(0,1,N)')); % varying frequency component

% create random system matrix
A = randn(N, N);
b = A*x_true + 0.8*rand(N,1); % add noise

% initial solution
x0 = A \ b;

%% Solve with unregularized CG
[x_star_CG, ~] = slv.CG(x0, A, b, 'niter', 100, ...
    'update_fun', @(itr,~,x_star,~)plot_fun(itr,sprintf('CG, iteration %d',itr),x_true,x_star));
fprintf('\tunregularized CG, norm error = %g\n', norm(x_star_CG - x_true)) % should be close to zero

%% Solve with L2-regularized CG
R = reg.L2(1e-2);
[x_star_L2CG, ~] = slv.CG(x0, A, b, 'niter', 100, 'R', R, ...
    'update_fun', @(itr,~,x_star,~)plot_fun(itr,sprintf('L2-CG, iteration %d',itr),x_true,x_star));
fprintf('\tL2-regularized CG, norm error = %g\n', norm(x_star_L2CG - x_true)) % should be close to zero

%% Solve with TV-regularized FISTA
R = reg.TV(1e-2);
[x_star_TVFISTA, ~] = slv.FISTA(x0, A, b, 'niter', 100, 'R', R, ...
    'update_fun', @(itr,~,x_star,~)plot_fun(sprintf('TV-FISTA, iteration %d',itr),x_true,x_star));
fprintf('\tTV-regularized FISTA, norm error = %g\n', norm(x_star_TVFISTA - x_true)) % should be close to zero

%% Solve with TV-regularized ADMM
R = reg.TV(1e-2);
[x_star_L1ADMM, ~] = slv.nlCG(x0, A, b, 'niter', 100, 'R', R, 't0', 1e-6,...
    'update_fun', @(itr,~,x_star,~)plot_fun(itr,sprintf('TV-ADMM, iteration %d',itr),x_true,x_star));
fprintf('\tTV-regularized ADMM, norm error = %g\n', norm(x_star_L1ADMM - x_true)) % should be close to zero

function plot_fun(itr,tit,x_gt,x)

    plot(x), hold on
    plot(x_gt,'-r','Linewidth',2), hold off
    axis off
    legend('estimate','truth')
    title(tit)
    ylim([min(x_gt(:))-0.5, max(x_gt(:))+0.5]);
    drawnow

end