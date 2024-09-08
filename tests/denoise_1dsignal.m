N = 1000;
x_true = 0.8*sin(2*pi*3.7*linspace(0,1,N)') + ... % low freq component
    0.3*sin(2*pi*21*linspace(0,1,N)') + ... % high freq component
    2*((1:N) > 500)'; % edge at N=500 (good for testing edge preservation)
A = randn(N, N);
b = A*x_true;
x0 = x_true + 0.8*rand(N,1);

%% Solve with unregularized CG
x_other = [x_true(:),x0(:)];
[x_star_CG, ~] = slv.CG(x0, A, b, 'niter', 100, ...
    'update_fun', @(itr,~,x_star,~)plot_fun(itr,x_other,x_star,'truth','initial','CG'));
fprintf('\tunregularized CG, norm error = %g\n', norm(x_star_CG - x_true)) % should be close to zero
x_other = [x_other,x_star_CG(:)];

%% Solve with L2-regularized CG
R = reg.L2(1e-2);
[x_star_L2CG, ~] = slv.CG(x0, A, b, 'niter', 100, 'R', R, ...
    'update_fun', @(itr,~,x_star,~)plot_fun(itr,x_other,x_star,'truth','initial','CG','L2-CG'));
fprintf('\tL2-regularized CG, norm error = %g\n', norm(x_star_L2CG - x_true)) % should be close to zero
x_other = [x_other,x_star_L2CG(:)];

%% Solve with TV-regularized FISTA
R = reg.TV(1e-2);
[x_star_TVFISTA, ~] = slv.FISTA(x0, A, b, 'niter', 100, 'R', R, ...
    'update_fun', @(itr,~,x_star,~)plot_fun(itr,x_other,x_star,'truth','initial','CG','L2-CG','TV-FISTA'));
fprintf('\tTV-regularized FISTA, norm error = %g\n', norm(x_star_TVFISTA - x_true)) % should be close to zero

function plot_fun(itr,x_other,x,varargin)

    plot(x_other), hold on
    plot(x), hold off
    legend(varargin{:})
    title(sprintf('iter %d', itr))
    drawnow

end