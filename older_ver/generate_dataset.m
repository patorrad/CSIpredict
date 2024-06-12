DISPLAY = true;                    % Whether to display the scene in a figure

% Enviornment params
SCENE_PATH = "/Users/vibhasathishkumar/matlab_dataset_generator/models/quarry.stl";
SCENE_SCALE = 1.0;                  % Scale such that 1 model unit = 1 meter

% Radio params
CENTER_FREQ = 2.408e9;              % Hz
BANDWIDTH = 40e6;                   % Hz
SUBCARRIERS = 128;                  % Number of subcarriers

% Tracing params
ANGULAR_SEPARATION = 1;            % low, medium or high (low means more rays)
SURFACE_MATERIAL = "concrete";      % brick, concrete, glass, metal, wood, etc.
MAX_REFLECTIONS = 3;                % Number of reflections to trace
MAX_DIFFRACTIONS = 1;               % Number of diffractions to trace
MAX_ABSOLUTE_PATH_LOSS = 120;       % Discard rays below this threshold (dBm)

% Sampling params
ANTENNA_HEIGHT = 1.0;               % How far above the ground the antenna is (meters)
TX_POSITION = [100; 170];           % Meters
RX_GRID_SPACING = 10.0;             % How far apart to sample the model surface (meters)
MAX_RAYS = 100;                     % Maximum number of rays to save in the dataset per RX
                                    % (janky and should be variable but idk how with hdf5)
% Output params
OUTPUT_PATH = "dataset.h5";




% Start measuring time
tic;

% Set up the environment
scene_mesh = stlread(SCENE_PATH);
viewer = siteviewer("SceneModel",scene_mesh,"ShowOrigin",false, "SceneModelScale", SCENE_SCALE);
viewer.Transparency = 1;

% Create the transmitter
tx = txsite("cartesian","AntennaPosition",place_on_ground(TX_POSITION, scene_mesh, ANTENNA_HEIGHT));
pm = propagationModel( ...
    "raytracing", ...
    "CoordinateSystem","cartesian", ...
    "SurfaceMaterial",SURFACE_MATERIAL, ...
    "MaxNumReflections", MAX_REFLECTIONS, ...
    "MaxNumDiffractions", MAX_DIFFRACTIONS, ...
    "AngularSeparation", ANGULAR_SEPARATION, ...
    "MaxAbsolutePathLoss", MAX_ABSOLUTE_PATH_LOSS ...
);

% Set up the RX positions
disp("Placing receivers...");
bounds = find_xy_bounds(scene_mesh);
x = bounds(1):RX_GRID_SPACING:bounds(2);
y = bounds(3):RX_GRID_SPACING:bounds(4);
rx_positions = zeros(3, length(x)*length(y));
for i = 1:length(x)
    for j = 1:length(y)
        rx_positions(:, (i-1)*length(y) + j) = place_on_ground([x(i); y(j)], scene_mesh, ANTENNA_HEIGHT);
        if mod((i-1)*length(y) + j, 100) == 0
            fprintf("Placed %d/%d receivers\n", (i-1)*length(y) + j, length(x)*length(y));
        end
    end
end

% Create the receivers
disp("Creating RX sites...")
disp("This step takes the longest for some reason...")
rxs = rxsite("cartesian","AntennaPosition",rx_positions);

if DISPLAY
    show(tx)
    show(rxs)
end

% Trace rays from TX to RX
disp("Tracing rays...")
rays = raytrace(tx,rxs,pm);


disp("Calculating signal data...")
csis_mag = zeros(length(rxs), SUBCARRIERS);
csis_phase = zeros(length(rxs), SUBCARRIERS);
ray_num = zeros(length(rxs), 1);
ray_delays = zeros(length(rxs), MAX_RAYS);
ray_path_losses = zeros(length(rxs), MAX_RAYS);
ray_aoas = zeros(length(rxs), 2, MAX_RAYS);
ray_aods = zeros(length(rxs), 2, MAX_RAYS);

for i = 1:length(rxs)
    % Get the rays for this receiver
    this_rays = rays{1, i};

    % Get the ray delays and amplitudes
    delays = [this_rays.PropagationDelay];
    amplitudes = dbm_to_watts(-[this_rays.PathLoss]);

    % Get the frequencies of all the subcarriers
    fc = (CENTER_FREQ - BANDWIDTH/2) + (BANDWIDTH/SUBCARRIERS * (0:SUBCARRIERS-1));

    % Calculate the channel freq response
    h = zeros(SUBCARRIERS,1);
    for s = 1:SUBCARRIERS
        h(s) = sum(amplitudes .* exp(-1j*2*pi*fc(s)*delays));
    end

    % Record the CSI
    csis_phase(i, :) = angle(h);
    csis_mag(i, :) = abs(h);

    % Record the other ray data
    if length(this_rays) > MAX_RAYS
        disp("Warning: More than MAX_RAYS rays detected. Consider increasing.")
        this_rays = this_rays(1:MAX_RAYS);
    end

    ray_num(i) = length(this_rays);
    if isempty(this_rays)
        continue
    end
    ray_delays(i, :) = pad_array([this_rays.PropagationDelay], MAX_RAYS);
    ray_path_losses(i, :) = pad_array([this_rays.PathLoss], MAX_RAYS);
    ray_aoas(i, :, :) = pad_array([this_rays.AngleOfArrival], MAX_RAYS);
    ray_aods(i, :, :) = pad_array([this_rays.AngleOfDeparture], MAX_RAYS);
end

% Save the data
disp("Saving data...")

% Delete the file if it already exists
if isfile(OUTPUT_PATH)
    disp("Warning: Output file already exists. Overwriting.")
    delete(OUTPUT_PATH);
end

% Write the CSI data
h5create(OUTPUT_PATH, '/csis_mag', size(csis_mag));
h5write(OUTPUT_PATH, '/csis_mag', csis_mag);
h5create(OUTPUT_PATH, '/csis_phase', size(csis_phase));
h5write(OUTPUT_PATH, '/csis_phase', csis_phase);

% Write the position
h5create(OUTPUT_PATH, '/positions', size(rx_positions.')); % Transposed
h5write(OUTPUT_PATH, '/positions', rx_positions.');

% Write the ray data
h5create(OUTPUT_PATH, '/ray_aoas', size(ray_aoas));
h5write(OUTPUT_PATH, '/ray_aoas', ray_aoas);
h5create(OUTPUT_PATH, '/ray_aods', size(ray_aods));
h5write(OUTPUT_PATH, '/ray_aods', ray_aods);
h5create(OUTPUT_PATH, '/ray_delays', size(ray_delays));
h5write(OUTPUT_PATH, '/ray_delays', ray_delays);
h5create(OUTPUT_PATH, '/ray_path_losses', size(ray_path_losses));
h5write(OUTPUT_PATH, '/ray_path_losses', ray_path_losses);

% Write the metadata
h5writeatt(OUTPUT_PATH, '/', 'created', datestr(now));
h5writeatt(OUTPUT_PATH, '/', 'scene_path', SCENE_PATH);
h5writeatt(OUTPUT_PATH, '/', 'scene_scale', SCENE_SCALE);
h5writeatt(OUTPUT_PATH, '/', 'center_freq', CENTER_FREQ);
h5writeatt(OUTPUT_PATH, '/', 'bandwidth', BANDWIDTH);
h5writeatt(OUTPUT_PATH, '/', 'subcarriers', SUBCARRIERS);
h5writeatt(OUTPUT_PATH, '/', 'angular_separation', ANGULAR_SEPARATION);
h5writeatt(OUTPUT_PATH, '/', 'surface_material', SURFACE_MATERIAL);
h5writeatt(OUTPUT_PATH, '/', 'max_reflections', MAX_REFLECTIONS);
h5writeatt(OUTPUT_PATH, '/', 'max_diffractions', MAX_DIFFRACTIONS);
h5writeatt(OUTPUT_PATH, '/', 'max_absolute_path_loss', MAX_ABSOLUTE_PATH_LOSS);
h5writeatt(OUTPUT_PATH, '/', 'antenna_height', ANTENNA_HEIGHT);
h5writeatt(OUTPUT_PATH, '/', 'tx_position', TX_POSITION);
h5writeatt(OUTPUT_PATH, '/', 'rx_grid_spacing', RX_GRID_SPACING);

% Print elapsed time
toc;
disp("Done!")

% Converts from dBm to Watts
function watts = dbm_to_watts(dbm)
    watts = 10.^((dbm-30)/10);
end

% Converts from Watts to dBm
function dbm = watts_to_dbm(watts)
    dbm = 10.*log10(watts) + 30;
end

% Finds the 3D position on the ground given a 2D position
function position = place_on_ground(pos_2d, scene_mesh, height_offset)
    % Create a ray from the position to the ground
    [intersect, ~, ~, ~, xcoor] = TriangleRayIntersection( ...
        [pos_2d; -1000], ...
        [0; 0; 1], ...
        scene_mesh.Points(scene_mesh.ConnectivityList(:, 1), :), ...
        scene_mesh.Points(scene_mesh.ConnectivityList(:, 2), :), ...
        scene_mesh.Points(scene_mesh.ConnectivityList(:, 3), :) ...
    );
    
    % Get the intersections
    xcoor = xcoor(intersect == 1, :, :);
    if isempty(xcoor)
        position = [pos_2d; 0];
        return
    end
    height = max(xcoor(:, 3)) + height_offset;
    position = [pos_2d; height];
end

% Finds the bounding box of the scene
function bounds = find_xy_bounds(scene_mesh)
    ep = 0.1;   % move the bounds in to avoid the edges
    bounds = [ ...
        min(scene_mesh.Points(:, 1)) + ep, ...
        max(scene_mesh.Points(:, 1)) - ep, ...
        min(scene_mesh.Points(:, 2)) + ep, ...
        max(scene_mesh.Points(:, 2)) - ep, ...
        min(scene_mesh.Points(:, 3)) + ep, ...
        max(scene_mesh.Points(:, 3)) - ep ...
    ];
end

% Pad an array with zeros to the specified size
function padded = pad_array(array, new_length)
    padded = zeros(size(array, 1), new_length);
    padded(:, 1:size(array, 2)) = array;
end
