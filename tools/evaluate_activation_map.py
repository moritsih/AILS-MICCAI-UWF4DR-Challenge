import numpy as np
import matplotlib.pyplot as plt

def create_custom_activation_map():
    # Create an activation map with values increasing by 0.04 for each line
    activation_values = np.arange(0, 1.04, 0.04)  # Values from 0 to 1 with step of 0.04
    activation_map = np.tile(activation_values, (25, 1))  # Repeat the values across 25 rows

    return activation_map

def apply_blue_to_red_colormap(activation_map):
    # Normalize the activation map to range [0, 1] (it is already in range but for safety)
    normalized_map = activation_map / np.max(activation_map)

    # Use matplotlib to apply the blue-to-red color map
    colormap = plt.get_cmap('coolwarm')  # 'coolwarm' goes from blue to red
    colored_map = colormap(normalized_map)

    # Convert to RGB format and scale to [0, 255]
    colored_map = (colored_map[:, :, :3] * 255).astype(np.uint8)
    
    return colored_map

# Create a custom activation map
activation_map = create_custom_activation_map()

# Apply the blue-to-red colormap
colored_map = apply_blue_to_red_colormap(activation_map)

# Display the custom color map
plt.imshow(colored_map)
plt.title('Custom Blue to Red Color Map')
plt.axis('off')
plt.show()