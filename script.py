import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
# import mahotas


# Reads the image index number as input and returns its object
def read_image(num_img: int) -> np.ndarray:
    path = f'./Img/image_name{num_img}.jpg'
    if os.path.exists(path):
        return cv2.imread(path)
    return None

# Displays the image passed as input and waits the window to be closed
def display_image(img: np.ndarray) -> None:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
    cv2.imshow("Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Gets the image object as input, treats it and returns a list of arrays which are histogram values for each of the three color channels
def rgb_histogram(image: np.ndarray, white_pixels: int) -> list:
    channels = cv2.split(image) # splits image in three color channels (rgb)
    list_rgb = [] 
    colors = ('r', 'g', 'b')
    for channel in channels:
        histogram = cv2.calcHist([channel], [0], None, [256], [0, 256]) # Calculates histogram values for channel given
        histogram[255]= histogram[255]-white_pixels # Removes white pixels from histogram values regarding background
        # eq_histogram = equalize_histogram(histogram)
        # list_rgb.append(eq_histogram)
        list_rgb.append(histogram) # Appends histogram channel to list containing all rgb ones
    return list_rgb # Returns list containing rgb channels

# Shows histogram of rgb channels with a matplotlib plot with three channels overlapped
def plot_hist(rgb_list: list) -> None:
    colors = ('r', 'g', 'b')
    for channel, color in zip(rgb_list, colors):
        plt.plot(channel, color)
    plt.xlabel('Intensity')
    plt.ylabel('Pixel Number')
    plt.title('RGB Color Histogram')
    plt.legend(['Red', 'Green', 'Blue'])
    plt.show()


# Equalize histogram given
def equalize_histogram(histogram: np.ndarray) -> np.ndarray:
    gray_hist = np.sum(histogram, axis=1) 
    gray_hist = gray_hist.astype(np.uint8)
    equalized_gray_hist = cv2.equalizeHist(gray_hist)
    equalized_gray_hist = equalized_gray_hist.astype(np.uint8)
    equalized_hist = np.zeros_like(histogram)
    equalized_hist[:, :] = equalized_gray_hist
    return equalized_hist

# Given an image object, returns the number of white pixels in it, useful to detect background size
def get_white_pixels(image: np.ndarray) -> int:
    num_white_pixels = np.count_nonzero(np.all(image == [255, 255, 255], axis=-1))
    return num_white_pixels

# calculate haralick texture features
# def haralick_features(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     features = mahotas.features.haralick(gray).mean(axis=0)
#     return np.array(features)


# Loads all images in dataset until end_num index and adds their data to main dictionary
def load_imgs() -> None:
    for num_img in range(end_num+1): # Creates main loop
        img1 = read_image(num_img) # Reads image with num_img index
        if img1 is not None: # Cheks if not null
            main_dict[num_img] = {} # Creates dictionary entry in main dictionary at num_img index
            rgb_list = rgb_histogram(img1, white_pixels) # Gets rgb color histograms for img given
            main_dict[num_img]['Histogram'] = rgb_list # Adds rgb histogram values arrays to 'Histogram' entry in main dictionary
            main_dict[num_img]['Texture'] = haralick_features(img1)
            

# Compares all images in main dictionary with image in selected_num_img and returns dictionary with similarity values
def compare_imgs(selected_num_img: int, comparing_dict: dict) -> dict:
    sel_hist_list = main_dict[selected_num_img]['Histogram'] # Gets histogram value from base image
    sel_texture = main_dict[selected_num_img]['Texture'] # Gets texture haralick value from base image
    del main_dict[selected_num_img] # Deletes entry from base image in main dictionary
    for key in main_dict.keys(): # For each key in dictionary
        comparing_dict[key] = get_comp_value(sel_hist_list, sel_texture, key) # Gets similarity value and adds it to similarity dictionary
    comparing_dict = dict(sorted(comparing_dict.items(), key = lambda x: x[1], reverse=True)) # Sorts dictionary by similarity
    return comparing_dict

# Gets similarity value between base image and key image
def get_comp_value(sel_hist_list: list, sel_texture, key: int) -> float:
    hist1 = np.array([sel_hist_list])
    hist2 = np.array([main_dict[key]['Histogram']])
    correlation = np.corrcoef(hist1.flatten(), hist2.flatten())[0, 1] # Calculates correlation between two histograms
    texture_similarity = np.linalg.norm(sel_texture-main_dict[key]['Texture']) # Calculates similarity in textures
    print(texture_similarity)
    return correlation # To change

# Displays num_top_values images (most similar)
def get_top_values(num_top_values: int) -> None:
    list_top = list(ordered_dict.keys())[:num_top_values]
    for img_num in list_top:
        display_image(read_image(img_num))



main_dict = {}  # Main dictionary with img data
end_num = 1484 # Last index number of images in memory

white_img = read_image(1373) # Creates image array of background removal model image
white_pixels = int(0.9*get_white_pixels(white_img)) # Gets white pixels number from white_img
load_imgs() # Loads all images data in main dictionary

selected_img = 7 # Image kits needes to be similar to
comparing_dict = {} # Comparing dictionary with similarity values
ordered_dict = compare_imgs(selected_img, comparing_dict) # Compares all images and gets similarity values in dictionary in order
get_top_values(10) # Get top values

