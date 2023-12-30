%% Face Detection and Tracking Using Live Video Acquisition
% This example shows how to automatically detect and track a face in a live
% video stream, using the KLT algorithm.   
% 
%   Copyright 2014 The MathWorks, Inc.

%% Overview
% Object detection and tracking are important in many computer vision
% applications including activity recognition, automotive safety, and
% surveillance.  In this example you will develop a simple system for
% tracking a single face in a live video stream captured by a webcam.
% MATLAB provides webcam support through a Hardware Support Package,
% which you will need to download and install in order to run this example. 
% The support package is available via the 
% <matlab:supportPackageInstaller Support Package Installer>.
%
% The face tracking system in this example can be in one of two modes:
% detection or tracking. In the detection mode you can use a
% |vision.CascadeObjectDetector| object to detect a face in the current
% frame. If a face is detected, then you must detect corner points on the 
% face, initialize a |vision.PointTracker| object, and then switch to the 
% tracking mode. 
%
% In the tracking mode, you must track the points using the point tracker.
% As you track the points, some of them will be lost because of occlusion. 
% If the number of points being tracked falls below a threshold, that means
% that the face is no longer being tracked. You must then switch back to the
% detection mode to try to re-acquire the face.

%% Setup
% Create objects for detecting faces, tracking points, acquiring and
% displaying video frames.

% Create the face detector object.
faceDetector = vision.CascadeObjectDetector();

% Create the point tracker object.
pointTracker = vision.PointTracker('MaxBidirectionalError', 2);

% Create the webcam object.
cam = webcam();

% Capture one frame to get its size.
videoFrame = snapshot(cam);
frameSize = size(videoFrame);

% Create the video player object. 
videoPlayer = vision.VideoPlayer('Position', [100 100 [frameSize(2), frameSize(1)]+30]);

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

runLoop = true;
numPts = 0;
frameCount = 0;

while runLoop
    
    % Get the next frame.
    videoFrame = snapshot(cam);
    videoFrameGray = im2gray(videoFrame);
    frameCount = frameCount + 1;
    
    if numPts < 10
        % Detection mode.
        bbox = faceDetector.step(videoFrameGray);
        
        if ~isempty(bbox)
            % Find corner points inside the detected region.
            points = detectMinEigenFeatures(videoFrameGray, 'ROI', bbox(1, :));
            
            % Re-initialize the point tracker.
            xyPoints = points.Location;
            numPts = size(xyPoints,1);
            release(pointTracker);
            initialize(pointTracker, xyPoints, videoFrameGray);
            
            % Save a copy of the points.
            oldPoints = xyPoints;
            
            % Convert the rectangle represented as [x, y, w, h] into an
            % M-by-2 matrix of [x,y] coordinates of the four corners. This
            % is needed to be able to transform the bounding box to display
            % the orientation of the face.
            bboxPoints = bbox2points(bbox(1, :));  
            
            % Convert the box corners into the [x1 y1 x2 y2 x3 y3 x4 y4] 
            % format required by insertShape.
            bboxPolygon = reshape(bboxPoints', 1, []);
            
            % Display a bounding box around the detected face.
            videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygon, 'LineWidth', 3);
            
            % Display detected corners.
            videoFrame = insertMarker(videoFrame, xyPoints, '+', 'MarkerColor', 'white');
        end
        
    else
        % Tracking mode.
        [xyPoints, isFound] = step(pointTracker, videoFrameGray);
        visiblePoints = xyPoints(isFound, :);
        oldInliers = oldPoints(isFound, :);
                
        numPts = size(visiblePoints, 1);       
        
        if numPts >= 10
            % Estimate the geometric transformation between the old points
            % and the new points.
            [xform, inlierIdx] = estgeotform2d(...
                oldInliers, visiblePoints, 'similarity', 'MaxDistance', 4);
            oldInliers    = oldInliers(inlierIdx, :);
            visiblePoints = visiblePoints(inlierIdx, :);
            
            % Apply the transformation to the bounding box.
            bboxPoints = transformPointsForward(xform, bboxPoints);
            
            % Convert the box corners into the [x1 y1 x2 y2 x3 y3 x4 y4] 
            % format required by insertShape.
            bboxPolygon = reshape(bboxPoints', 1, []);
    
% Code by us
% --------------------------------------------------------------------------------------------
            % Create a meshgrid of coordinates for the image
            [xq, yq] = meshgrid(1:size(videoFrameGray, 2), 1:size(videoFrameGray, 1));
            
            % Check which points in the meshgrid are inside the polygon
            insidePolygon = inpolygon(xq(:), yq(:), bboxPoints(:, 1), bboxPoints(:, 2));
            
            %access the modified RBG-Values
            finalRGB = getappdata(slider, 'finalRGB');

            % Flip the pixels manually for each color channel
            for channel = 1:3  % assuming RGB image
                channelPixels = videoFrame(:, :, channel);
                if channel == 1
                channelPixels(insidePolygon) = channelPixels(insidePolygon) + finalRGB(1);
                end
                if channel == 2
                channelPixels(insidePolygon) = channelPixels(insidePolygon) + finalRGB(2);
                end
                if channel == 3
                channelPixels(insidePolygon) = channelPixels(insidePolygon) + finalRGB(3);
                end
                videoFrame(:, :, channel) = channelPixels;
            end

            videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygon, 'LineWidth', 3);
            
            % Reset the points.
            oldPoints = visiblePoints;
            setPoints(pointTracker, oldPoints);
        end
% --------------------------------------------------------------------------------------------
%End Code by us
    end
        
    % Display the annotated video frame using the video player object.
    step(videoPlayer, videoFrame);

    % Check whether the video player window has been closed.
    runLoop = isOpen(videoPlayer);
end

% Clean up.
clear cam;
release(videoPlayer);
release(pointTracker);
release(faceDetector);

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
