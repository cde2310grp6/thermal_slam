import matplotlib.pyplot as plt
import numpy as np
import math
import time
from scipy.interpolate import interp1d  # For interpolation
import matplotlib.colors as mcolors  # Correct import for mcolors
from PIL import Image  # Import Pillow for image processing

flag = 1

class Robot:
    def __init__(self, position, pose, fov):
        self.position = position  # [y, x]
        self.pose = pose  # Orientation in degrees
        self.fov = fov  # Field of View in degrees

# Function to load an image and convert it to a grid
def load_image_to_grid(image_path):
    # Open the image and convert to grayscale
    img = Image.open(image_path).convert('L')  # 'L' mode is for grayscale
    
    # Convert image data to a numpy array
    img_data = np.array(img)
    
    # Map pixel values to 0 (white) and -1 (black)
    # White pixels (255) become 0, black pixels (0) become -1
    grid = np.where(img_data == 255, 0, -1)  # 0 for white, -1 for black
    
    return grid

def load_heat_map(image_path):
    # Open the image and convert to grayscale
    img = Image.open(image_path).convert('RGB')
    
    # Convert image data to a numpy array
    img_data = np.array(img)

    y = len(img_data)
    x = len(img_data[0])
    heat = np.zeros((y,x))
    for i in range(y):
        for j in range(x):
            if (img_data[i][j] == [255,255,255]).all():
                heat[i][j] = 0
            else:
                heat[i][j] = img_data[i][j][0]
                print(heat[i][j])
    return heat 

# Load the image and convert it to a grid
grid = load_image_to_grid("map.png")  # Replace with your image file path
heat = load_heat_map("test.png")
print(heat)

# Simulated 1D thermal readings (corresponding to rays)
data = np.zeros(32) # Example temperature readings

# Robot's initial position and properties
bot = Robot(position=[50, 25], pose=0, fov=60)

def ReadWall(bot, data):
    rows, cols = len(heat), len(heat[0])
    cx, cy = bot.position  # Robot's position (y, x)
    start_angle = bot.pose - bot.fov / 2  
    end_angle = bot.pose + bot.fov / 2  

    # Increase the number of rays based on the FOV and resolution
    num_rays = len(data)  # Number of rays to cast based on FOV (higher value = finer resolution)
    
    # Apply interpolation to the thermal data
    angles = np.linspace(start_angle, end_angle, num_rays)  # More rays are now cast

    for idx, angle in enumerate(angles):
        rad_angle = math.radians(angle)
        dx = math.cos(rad_angle)
        dy = math.sin(rad_angle)

        for step in range(1, 300):  # Adjust range if needed
            # Compute the grid position for the ray
            x_pos = int(round(cx + step * dy))
            y_pos = int(round(cy + step * dx))

            # Check if the position is within bounds
            if 0 <= x_pos < rows and 0 <= y_pos < cols:
                # If the grid cell is not empty (value is not 0), stop the ray and apply the thermal data
                if heat[x_pos][y_pos] != 0:  # Wall detection
                    data[idx] = heat[x_pos][y_pos]
                    break  # Stop at the first non-zero value (wall)
            else:
                break  # Stop the ray if it goes out of bounds

def PaintWall(bot, data):
    rows, cols = len(grid), len(grid[0])
    cx, cy = bot.position  # Robot's position (y, x)
    
    start_angle = bot.pose - bot.fov / 2  
    end_angle = bot.pose + bot.fov / 2  

    # Increase the number of rays based on the FOV and resolution
    num_rays = int(bot.fov * 2)  # Number of rays to cast based on FOV (higher value = finer resolution)
    
    # Apply interpolation to the thermal data
    x = np.linspace(0, len(data) - 1, len(data))  # Indices of the input data
    interpolator = interp1d(x, data, kind='linear', fill_value='extrapolate')  # Linear interpolation
    interpolated_data = interpolator(np.linspace(0, len(data) - 1, num_rays))  # Interpolated data

    angles = np.linspace(start_angle, end_angle, num_rays)  # More rays are now cast

    for idx, angle in enumerate(angles):
        rad_angle = math.radians(angle)
        dx = math.cos(rad_angle)
        dy = math.sin(rad_angle)

        for step in range(1, 300):  # Adjust range if needed
            # Compute the grid position for the ray
            x_pos = int(round(cx + step * dy))
            y_pos = int(round(cy + step * dx))

            # Check if the position is within bounds
            if 0 <= x_pos < rows and 0 <= y_pos < cols:
                # If the grid cell is not empty (value is not 0), stop the ray and apply the thermal data
                if grid[x_pos][y_pos] != 0:  # Wall detection
                    if grid[x_pos][y_pos] == -1:
                        grid[x_pos][y_pos] = interpolated_data[idx]
                    else:
                        grid[x_pos][y_pos] = (grid[x_pos][y_pos] + interpolated_data[idx]) / 2
                    break  # Stop at the first non-zero value (wall)
            else:
                break  # Stop the ray if it goes out of bounds

plt.ion()  # turning interactive mode on

# Apply the function
while(True):
    ReadWall(bot, data)
    PaintWall(bot, data)
    if bot.pose >= 180:
        bot.pose = -180
    bot.pose += 10
    if bot.position[0] >=60 or bot.position[0] <= 30:
        flag = -flag
    bot.position[0] += flag



    # Mask the empty spaces (0 values) to exclude them from the normalization
    masked_grid = np.ma.masked_equal(grid, 0)

    # Normalize based on temperature range (min/max values from the masked grid)
    norm = mcolors.Normalize(vmin=masked_grid.min(), vmax=masked_grid.max())

    # Clear the previous plot
    plt.clf()

    # Plot the heatmap
    plt.imshow(masked_grid, cmap='magma', norm=norm)
    plt.colorbar(label="Temperature")

    # Plot the robot's position
    robot_y, robot_x = bot.position
    plt.scatter(robot_x, robot_y, color='cyan', s=100, marker='x', label="Robot")  # Robot marker

    # Add FOV lines
    start_angle = bot.pose - bot.fov / 2
    end_angle = bot.pose + bot.fov / 2
    fov_angles = np.linspace(start_angle, end_angle, 2)  # Two lines at start and end angles
    for angle in fov_angles:
        rad_angle = math.radians(angle)
        dy = math.cos(rad_angle)
        dx = math.sin(rad_angle)

        # Extend the line from the robot position to a point far in the direction of the angle
        line_length = 15  # How far to extend the FOV line
        x_end = int(round(robot_x + line_length * dy))
        y_end = int(round(robot_y + line_length * dx))

        plt.plot([robot_x, x_end], [robot_y, y_end], color='black', linestyle='--', label="FOV Line" if angle == start_angle else "")

    # Draw the updated plot
    plt.draw()
    plt.pause(0.1)  # Pause to allow the plot to update
    time.sleep(0.05)
