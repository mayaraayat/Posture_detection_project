import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import pickle
import cv2

with open('data.json') as f:
    data = json.load(f)

try:
    with open('chairpos', 'rb') as f:
        poslist = pickle.load(f)
except:
    poslist = []


# Load the chair image
chair_image = cv2.imread('./static/images/chair01.jpg')

# Define the polygon coordinates
print([a[1] for a in enumerate(poslist)])
polygon_pts = np.array([a[1] for a in enumerate(poslist)][0], np.int32).reshape((-1,1,2))
print(polygon_pts)
# Create an empty mask and fill it with the polygon area
mask = np.zeros((chair_image.shape[0], chair_image.shape[1]), dtype=np.uint8)


cv2.fillPoly(mask, [polygon_pts], 255)
'''plt.imshow(mask, cmap='gray')
plt.title('Mask Visualization')
plt.show()'''


# Create a figure for the animation
fig, ax = plt.subplots()
im = ax.imshow(np.zeros_like((chair_image[:, :, 0])), cmap='gray')
pressure_data = data.get('pressureData', [])
pressure_matrices = [entry["pressureMatrix"] for entry in pressure_data]


def update(frame):
    ax.clear()
    im = ax.imshow(pressure_matrices[frame], cmap='hot')
    return im

ani = FuncAnimation(fig, update, frames=len(pressure_matrices), interval=0)
plt.show()
# Convert the FuncAnimation to an OpenCV-compatible format
fig.canvas.draw()
animation_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
animation_img = animation_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
animation_img = cv2.cvtColor(animation_img, cv2.COLOR_RGB2BGR)
# Apply the mask to the animation image

masked_animation = cv2.bitwise_and(animation_img, animation_img, mask=mask)

# Overlay the masked animation onto the chair image within the defined polygon
result = chair_image.copy()
result[np.where(masked_animation != 0)] = masked_animation[np.where(masked_animation != 0)]

# Show the resulting image
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()




