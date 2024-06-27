# Football Kit CBIR (Content-Based Image Retrieval)

## Overview
This project implements a Content-Based Image Retrieval (CBIR) system for football kits. It allows users to find similar football kits based on color and texture features.

## Motivation
In the world of football, fans often express their passion through team kits. This CBIR application simplifies the process of finding similar kits by using image processing techniques to match color palettes and textures.

## Features
- RGB color histogram analysis
- Haralick texture feature extraction
- Similarity comparison using histogram correlation and Euclidean distance
- Weighted scoring system (90% color, 10% texture)
- Background removal for accurate comparisons

## Technologies Used
- Python
- OpenCV
- Pandas
- NumPy
- Mahotas (for Haralick texture features)

## Dataset
The project uses a dataset from Kaggle containing over 1400 high-quality images of football kits with white backgrounds.

## Implementation Details
1. Data Preparation:
   - Download images from provided Excel file links
   - Store images in "Img" folder

2. Image Processing:
   - Background removal
   - RGB histogram extraction
   - Haralick texture feature extraction

3. Similarity Calculation:
   - Histogram correlation for color similarity
   - Euclidean distance for texture similarity
   - Weighted average of color and texture scores

4. Results:
   - Display top X most similar kits to the user

## How to Run
1. Ensure you have Python installed with the required libraries (OpenCV, Pandas, NumPy, Mahotas)
2. Place the dataset in the correct folder (or update the path in the code)
3. Run the main script (details to be provided)

Note: The code was developed and tested in Google Colab due to package installation requirements.

## Future Improvements
- Implement histogram normalization for better performance
- Optimize for larger datasets and lower RAM usage

## Contributors
- Federico Paschetta
- Cecilia Peccolo
- Nicola Maria D'Angelo

## Acknowledgments
- Universidad Politécnica de Madrid
- Kaggle for providing the dataset
