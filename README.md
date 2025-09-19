Ego-Vehicle Trajectory Estimation

This project estimates the ego-vehicle’s trajectory from stereo camera data using the traffic light as a fixed world reference. The output is a bird’s-eye view (BEV) plot of the vehicle’s path.

Setup

I first created a virtual environment using Python 3.13.7 and installed dependencies from requirements.txt:

python3.13 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

Game Plan

Before starting, I wrote a small plan outlining how to:

Estimate the car’s motion

Handle noise in bounding box detections

Compute a stable trajectory

Methodology
1. Bounding Box Center Extraction

I load dataset/bbox_light.csv which contains the traffic light bounding box per frame.

For each frame, I compute the center pixel (u, v) as:

u = (xmin + xmax) / 2, v = (ymin + ymax) / 2

2. Smoothing the Traffic Light Centers

Raw centers can jitter due to detector noise.

I smooth them using a Kalman filter, which fills missing frames (NaN values), reduces outliers, produces smoother motion than a simple rolling average

As a first pass, I also take a rolling mean over the last 5 frames to reduce high-frequency noise before feeding into the Kalman filter.

3. 3D Position from Depth Data

For each frame, I load the corresponding xyz/frame_XXXX.npz file, which stores a (H, W, 3) point cloud.

I grab the (X, Y, Z) point at the smoothed (u, v) center — or average a small patch around it for robustness.

These are in camera coordinates, where:

+X = forward

+Y = right

+Z = up

4. Ego-Vehicle Motion Estimation

I treat the traffic light position in the first frame as the world origin.

As the car moves, the traffic light appears to move in the camera frame.

I invert this motion to estimate ego-vehicle translation (x, y) over time, projected onto the ground plane.

5. Velocity Calculation

I compute velocity by differentiating the trajectory with respect to time.

Smoothing is applied again to reduce spurious spikes in velocity due to noise.

6. Visualization

Finally, I plot the estimated path in trajectory.png:

X-axis: forward distance

Y-axis: lateral distance

The result is a scatter plot of discrete car positions over time.

Output

trajectory.png – Bird’s-eye view plot of the car’s trajectory# wisconsin-autonomous-perception
