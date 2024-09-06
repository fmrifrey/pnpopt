%% Set up the problem
N = 128; % image size
nc = 8; % number of coils
R = 2; % acceleration (undersampling) factor
noise_fac = 0.2; % noise factor

% create the ground truth image
x = phantom(N); % N x N shepp logan phantom
RMSE_gt = @(x_est) sqrt(mean(vec(x - x_est).^2));

% simulate a sensitivity map
smap = mri_sensemap_sim('nx',N,'ncoil',nc);

% create an (undersampled) spiral sampling pattern
N_samples = ceil(N/R*1e3);
r = linspace(0,pi,N_samples)';
theta = linspace(0,N/R*pi,N_samples)';
omega = r.*[cos(theta),sin(theta)];

% create the system operator
nufft_args = {omega, [N,N], [6,6], 2*[N,N], [N/2,N/2], 'table', 2^10, 'minmax:kb'};
F = Gnufft(true(N), nufft_args); % NUFFT
w = ir_mri_density_comp(omega,'pipe','G',F); % density compensation
FS = Asense(F,smap); % sensitivity encoding

% simulate data (inversely criminal)
y = FS * x;
noise = noise_fac*mean(abs(y(:))) * (rand(N_samples,nc) + 1i*rand(N_samples,nc));
y = y + noise;

%% estimate the initial reconstruction
x0 = FS' * (w.*y); % with density compensation for initialization
x0 = ir_wls_init_scale(FS, y, x0); % fix scale

figure
imagesc(abs(x0)); axis off
title(sprintf('initial estimate\nRMSE = %g',RMSE_gt(x0)));

%% solve with unregularized CG
niter = 20;
[x_star,cost] = slv.CG(x0,FS,y,'niter',niter);

figure
subplot(2,1,1)
imagesc(abs(x_star)); axis off
title(sprintf('Unregularized CG\n(%d iterations)\nRMSE = %g',niter,...
    RMSE_gt(x_star)));
subplot(2,1,2)
plot(0:niter,cost); xlabel('iteration #'); ylabel('cost');
drawnow

%% solve with TV-nlCG
niter = 20;
R = reg.TV(1e-3,'l1'); % spatial total variation (no dim specified)
[x_star,cost] = slv.nlCG(x0,FS,y,'niter',niter,'R',R);

figure
subplot(2,1,1)
imagesc(abs(x_star)); axis off
title(sprintf('TV-regularized nlCG\n(%d iterations)\nRMSE = %g',niter,...
    RMSE_gt(x_star)));
subplot(2,1,2)
plot(0:niter,cost); xlabel('iteration #'); ylabel('cost');
drawnow

%% solve with TV-FISTA 
niter = 20;
R = reg.TV(1e-3,'l1'); % spatial total variation (no dim specified)
[x_star,cost] = slv.FISTA(x0,FS,y,'niter',niter,'R',R);

figure
subplot(2,1,1)
imagesc(abs(x_star)); axis off
title(sprintf('TV-FISTA\n(%d iterations)\nRMSE = %g',niter,...
    RMSE_gt(x_star)));
subplot(2,1,2)
plot(0:niter,cost); xlabel('iteration #'); ylabel('cost');
drawnow