clear; close all; clc;
% Code gives the Faraday threshold, the Gamma needed to get a certain
% Gamma/GammaF ratio and the required minimum radius to meet the 
% system requirements
%% Physical parameters
rho    = 949.0;        % [kg/m^3]
sigma  = 20.6e-3;      % [N/m]
nu     = 20.0e-6;      % [m^2/s]
g      = 9.81;         % [m/s^2]
omega0 = 160.0*pi;     % [rad/s]

%% Requirement parameter
delta_lambda = 0.10;   % require lambda_F / r_horizon <= delta_lambda

%% Gamma ratio target

Gamma_ratio_target = 0.95;   % [-]

%% Search interval for k
k_min = 500.0;         % [1/m]
k_max = 3000.0;        % [1/m]

%% Definitions
omegaF = omega0/2.0;

omega2  = @(k) g.*k + (sigma/rho).*k.^3;
omega   = @(k) sqrt(omega2(k));
gamma   = @(k) 2.0*nu.*k.^2;
omegag2 = @(k) g.*k;

Gamma_curve = @(k) ...
    (2.0 ./ omegag2(k)) .* sqrt( ...
    (omega2(k) + gamma(k).^2 - 0.25*omega0^2).^2 ...
    + omega0^2 .* gamma(k).^2 );

%% Faraday threshold from neutral curve
% The neutral curve is used here only to obtain GammaF.  The k-location of
% the minimum is not used as the Faraday wavelength in the requirement.
opts = optimset('TolX', 1e-12, 'Display', 'off');
[k_min_neutral, GammaF] = fminbnd(Gamma_curve, k_min, k_max, opts);

%% Desired forcing amplitude from selected Gamma/GammaF ratio
Gamma_target = Gamma_ratio_target * GammaF;   

%% Resonance wavenumber: omega(k) = omega0/2
% The Faraday wavelength used in the length requirement is obtained from the
% resonance condition, not from the minimum of the neutral curve.
residual = @(k) omega2(k) - omegaF^2;
kF_resonance = fzero(residual, [k_min, k_max]);

lambdaF_resonance = 2*pi/kF_resonance;

%% Radius requirement
r_horizon_min = lambdaF_resonance / delta_lambda;

%% Output
fprintf('\nFaraday threshold and resonance calculation\n');
fprintf('------------------------------------------\n');

fprintf('From neutral curve:\n');
fprintf('  GammaF                = %.10f [-]\n', GammaF);

fprintf('\nSelected forcing ratio:\n');
fprintf('  Gamma/GammaF target             = %.10f [-]\n', Gamma_ratio_target);
fprintf('  Required Gamma                  = %.10f [-]\n', Gamma_target);


fprintf('\nFrom resonance omega(k)=omega0/2:\n');
fprintf('  kF_resonance          = %.10f 1/m\n', kF_resonance);
fprintf('  lambdaF_resonance     = %.10e m\n', lambdaF_resonance);

fprintf('\nLength requirement, using lambdaF_resonance:\n');
fprintf('  r_horizon >= %.10e m = %.4f lambda_F\n', ...
    r_horizon_min, r_horizon_min/lambdaF_resonance);

%% Plot neutral curve
Nk = 3000;
k_plot = linspace(k_min, k_max, Nk);
Gamma_plot = Gamma_curve(k_plot);

% Plot in cm^{-1}, matching the visual convention used in Milewski et al.
k_plot_cm = k_plot/100.0;
k_min_neutral_cm = k_min_neutral/100.0;

figure('Color','w');

plot(k_plot_cm, Gamma_plot, 'k-', 'LineWidth', 3.0);
hold on;

% Horizontal dotted threshold line through the minimum, as in the Milewski
% neutral-curve figure. This visually marks GammaF, while the length
% requirement still uses kF_resonance from omega(kF)=omega0/2.
yline(GammaF, ':', 'Color', [0.45 0.45 0.45], ...
    'LineWidth', 2.5, 'HandleVisibility', 'off');

% Small marker at the minimum, for visual reference only.
plot(k_min_neutral_cm, GammaF, 'kx', ...
    'MarkerSize', 12, 'LineWidth', 1.5, 'HandleVisibility', 'off');

grid on; box on;

xlabel('$k$ [cm$^{-1}$]', 'Interpreter','latex', 'FontSize', 14);
ylabel('$\Gamma$', 'Interpreter','latex', 'FontSize', 14);

legend({'$\Gamma(k)$'}, ...
    'Interpreter','latex', ...
    'Location','best');

set(gca, 'FontSize', 12);
