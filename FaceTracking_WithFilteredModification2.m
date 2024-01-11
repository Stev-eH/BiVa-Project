% Webcam initialisieren
cam = webcam;

% Gesichtserkennungskaskade laden (benötigt die Computer Vision Toolbox)
faceDetector = vision.CascadeObjectDetector();

% Punkttracker für Augenpositionen erstellen
tracker = MultiObjectTrackerKLT;

% Capture one frame to get its size.
frame = snapshot(cam);
frameSize = size(frame);

% Create the video player object.
videoPlayer = vision.VideoPlayer('Position', [100 100 [frameSize(2), frameSize(1)]+30]);

%% Iterate until we have successfully detected a face
bboxes = [];
while isempty(bboxes)
    framergb = snapshot(cam);
    frame = rgb2gray(framergb);
    bboxes = faceDetector.step(frame);
end
tracker.addDetections(frame, bboxes);

% Code by us
% --------------------------------------------------------------------------------------------
% Creat a Slider
% Create a figure
slider = figure('Position', [100, 100, 400, 200], 'Name', 'RGB Sliders');

% Initial RGB values
initialRGB = [0, 0, 0]; % Adjust this as needed

% Create sliders for R, G, and B components
sliderR = uicontrol('Style', 'slider', 'Min', -255, 'Max', 255, 'Value', initialRGB(1), ...
    'Units', 'normalized', 'Position', [0.1, 0.7, 0.8, 0.1], 'Callback', @sliderCallback);

sliderG = uicontrol('Style', 'slider', 'Min', -255, 'Max', 255, 'Value', initialRGB(2), ...
    'Units', 'normalized', 'Position', [0.1, 0.5, 0.8, 0.1], 'Callback', @sliderCallback);

sliderB = uicontrol('Style', 'slider', 'Min', -255, 'Max', 255, 'Value', initialRGB(3), ...
    'Units', 'normalized', 'Position', [0.1, 0.3, 0.8, 0.1], 'Callback', @sliderCallback);

% Create a text box to display RGB values
textLabel = uicontrol('Style', 'text', 'String', 'RGB Values: [0, 0, 0]', ...
    'Units', 'normalized', 'Position', [0.1, 0.1, 0.8, 0.1]);

% Set unique tags for each slider
set(sliderR, 'Tag', 'sliderR');
set(sliderG, 'Tag', 'sliderG');
set(sliderB, 'Tag', 'sliderB');

% Store textLabel in app data
setappdata(slider, 'textLabel', textLabel);
setappdata(slider, 'finalRGB', initialRGB);
% --------------------------------------------------------------------------------------------
%End Code by us

%% Detection and Tracking
% Capture and process video frames from the webcam in a loop to detect and
% track a face. The loop will run for 400 frames or until the video player
% window is closed.

camOpen = true;
guiOpen = true;
numPts = 0;
frameNumber = 0;

% Endlosschleife für die Live-Videoverarbeitung
while camOpen && guiOpen
    % Get the next frame.
    framergb = snapshot(cam);
    frame = rgb2gray(framergb);

    if mod(frameNumber, 100) == 0
        % Detection mode.
        % (Re)detect faces.
        %
        % NOTE: face detection is more expensive than imresize; we can
        % speed up the implementation by reacquiring faces using a
        % downsampled frame:
        % bboxes = faceDetector.step(frame);
        bboxes = 2 * faceDetector.step(imresize(frame, 0.5));

        % Wenn ein Gesicht erkannt wurde
        if ~isempty(bboxes)
            tracker.addDetections(frame, bboxes);

            % Augenpositionen für den Punkttracker aus den BoundingBox-Koordinaten extrahieren
            eyes = [bboxes(1, 1) + bboxes(1, 3)/4, bboxes(1, 2) + bboxes(1, 4)/4;...
                bboxes(1, 1) + 3*bboxes(1, 3)/4, bboxes(1, 2) + bboxes(1, 4)/4];

            % Koordinaten für das Oval berechnen
            x = bboxes(1, 1);
            y = bboxes(1, 2);
            width = bboxes(1, 3);
            height = bboxes(1, 4);

            % Faktor für die Ovalform
            ovalFactor = 0.8;

            % Manuell ein gefülltes Oval um das Gesicht zeichnen
            ellipseVertices = ellipseToPolygon(x + width/2, y + height/2, width*ovalFactor, height, 100);
%            framergb = insertShape(framergb, 'FilledPolygon', ellipseVertices(i, :, :), 'Color', 'red');
        end
    else
% Tracking mode.
         % Track faces
        tracker.track(frame);
        numFoundboxes = size(tracker.BoxIds, 1);
        ellipseVertices = zeros(100, 2, numFoundboxes);
     
           

        for i = 1:numFoundboxes
            x = tracker.Bboxes(i, 1);
            y = tracker.Bboxes(i, 2);
            width = tracker.Bboxes(i, 3);
            height = tracker.Bboxes(i, 4);
            ellipseVertices(:, :, i) = ellipseToPolygon(x + width/2, y + height/2, width*ovalFactor, height, 100);
            framergb = insertShape(framergb, 'Polygon', ellipseVertices(:, :, i), 'LineWidth', 1);
        end

    
% Code by us
% --------------------------------------------------------------------------------------------
% Create a meshgrid of coordinates for the image
[xq, yq] = meshgrid(1:size(frame, 2), 1:size(frame, 1));

% Initialize insidePolygon
insidePolygon = false(size(frame, 1) * size(frame, 2), 1);

% Check which points in the meshgrid are inside the polygons
for boxIndex = 1:numFoundboxes
    insidePolygonBox = isPointInsidePolygon(xq(:), yq(:), ellipseVertices(:, 1, boxIndex), ellipseVertices(:, 2, boxIndex));
    insidePolygon = or(insidePolygon, insidePolygonBox);
end
            
            %access the modified RBG-Values
            finalRGB = getappdata(slider, 'finalRGB');

            % Flip the pixels manually for each color channel
            for channel = 1:3  % assuming RGB image
                channelPixels = framergb(:, :, channel);
                if channel == 1
                channelPixels(insidePolygon) = channelPixels(insidePolygon) + finalRGB(1);
                end
                if channel == 2
                channelPixels(insidePolygon) = channelPixels(insidePolygon) + finalRGB(2);
                end
                if channel == 3
                channelPixels(insidePolygon) = channelPixels(insidePolygon) + finalRGB(3);
                end
                framergb(:, :, channel) = channelPixels;
            end

 %           framergb = insertShape(framergb, 'Polygon', ellipseVertices, 'LineWidth', 1);
            
% --------------------------------------------------------------------------------------------
%End Code by us
    end
        % Flip image
        framergb = fliplr(framergb);
        frameNumber = frameNumber + 1;

        % Display the annotated video frame using the video player object.
        step(videoPlayer, framergb);

        % Check whether the video player window has been closed.
        camOpen = isOpen(videoPlayer);
        guiOpen = ~isempty(findobj('name','RGB Sliders'));

end

    % Clean up.
    clear cam;
    clear slider;
    release(videoPlayer);
    release(faceDetector);
    delete(findall(0));


    % Funktion, um Ellipsenpunkte zu berechnen
    function vertices = ellipseToPolygon(centerX, centerY, width, height, numPoints)
    theta = linspace(0, 2*pi, numPoints);
    x = centerX + width/2 * cos(theta);
    y = centerY + height/2 * sin(theta);
    vertices = [x', y'];
    end
    % Code by us
% --------------------------------------------------------------------------------------------
% Callback function for the sliders
function sliderCallback(src, ~)
    % Access the figure handle from the source
    slider = ancestor(src, 'figure');
    % Access textLabel from app data
    textLabel = getappdata(slider, 'textLabel');
    
    % Access the sliders' handles using findobj
    sliderR = findobj(slider, 'Tag', 'sliderR');
    sliderG = findobj(slider, 'Tag', 'sliderG');
    sliderB = findobj(slider, 'Tag', 'sliderB');

    redValue = get(sliderR, 'Value');
    greenValue = get(sliderG, 'Value');
    blueValue = get(sliderB, 'Value');
    
    % Update the displayed RGB values
    set(textLabel, 'String', ['RGB Values: [', num2str(redValue), ', ', num2str(greenValue), ', ', num2str(blueValue), ']']);
    setappdata(slider, 'finalRGB', [redValue, greenValue, blueValue]);
    
end

function outputMatrix = isPointInsidePolygon(x, y, polyX, polyY)
    % Assuming polyX and polyY are arrays of x and y coordinates of the polygon vertices
    numVertices = length(polyX);
    
    % Initialize the insideMatrix with false values
    insideMatrix = false(size(x));

    for i = 1:numVertices
        x1 = polyX(i);
        y1 = polyY(i);
        x2 = polyX(mod(i, numVertices) + 1);
        y2 = polyY(mod(i, numVertices) + 1);

        % Check if the ray crosses the edge
        insideEdge = ((y1 > y) ~= (y2 > y)) & (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1);

        % Toggle the inside/outside status for points on the edge
        insideMatrix = xor(insideMatrix, insideEdge);

        outputMatrix = insideMatrix;
    end
end
% --------------------------------------------------------------------------------------------
%End Code by us

%% References
% Viola, Paul A. and Jones, Michael J. "Rapid Object Detection using a
% Boosted Cascade of Simple Features", IEEE CVPR, 2001.
%
% Bruce D. Lucas and Takeo Kanade. An Iterative Image Registration 
% Technique with an Application to Stereo Vision. 
% International Joint Conference on Artificial Intelligence, 1981.
%
% Carlo Tomasi and Takeo Kanade. Detection and Tracking of Point Features. 
% Carnegie Mellon University Technical Report CMU-CS-91-132, 1991.
%
% Jianbo Shi and Carlo Tomasi. Good Features to Track. 
% IEEE Conference on Computer Vision and Pattern Recognition, 1994.
%
% Zdenek Kalal, Krystian Mikolajczyk and Jiri Matas. Forward-Backward
% Error: Automatic Detection of Tracking Failures.
% International Conference on Pattern Recognition, 2010

