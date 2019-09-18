"""
Created on Wed Sep 11 10:39:48 2019
@author: varun
TASKS-
1. LBP feature extraction for each 100X100 pixel window & concatenate results of given image ID.
2. LBP feature extraction for all images in given folder path.
3. Finding 'k' most similar images to given image ID based on LBP feature descriptors.
"""

import os
from os import path
from sys import exit
import numpy as np
import cv2 as cv
from skimage import feature
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import re
import csv
import time
import sys


# dictionary to store feature descriptors
# format = {image_id -> [histogram array, histogram bin edges, image file path]}
image_lbp_dict = {}
# image filename = IMAGE_PREFIX + image_id + IMAGE_SUFFIX
IMAGE_PREFIX = '/Hand_'
IMAGE_SUFFIX = '.jpg'
# Block size parameters
BLOCK_HEIGHT = 100
BLOCK_WIDTH = 100
# LBP parameters
LBP_RADIUS = 1  # TODO: TEST varying radii
LBP_NOP = 8 * LBP_RADIUS
LBP_METHOD = 'default'  # TODO: TEST 'uniform'
# Histogram parameters
HIST_BINS = 2 ** LBP_NOP  # TODO: TEST values for 'uniform'
HIST_RANGE = (0, int(2 ** LBP_NOP))
# Output parameters
# Sample - /home/varun/Desktop/MWDB/Project/output/
OUTPUT_PATH = str(input('Enter the output folder to store all results: ') + '/')
OUTPUT_LBP_FOLDER_FILENAME = 'lbp_folder_' + time.strftime('%m%d%Y') + '_' + time.strftime('%H%M%S') + '.csv'
OUTPUT_LBP_IMAGE_FILENAME = '_lbp_image_' + time.strftime('%m%d%Y') + '_' + time.strftime('%H%M%S') + '.csv'
OUTPUT_LBP_HISTOGRAM_FILENAME = '_lbp_histogram_' + time.strftime('%m%d%Y') + '_' + time.strftime('%H%M%S') + '.png'


# Returns image file path- FOLDER path + PREFIX + ID + SUFFIX
# arguments - folder path, image ID
def get_path_from_id(dataset_path, image_id):
    return str(dataset_path + IMAGE_PREFIX + image_id + IMAGE_SUFFIX)


# loads original image -> converts to gray scale for LBP
# arguments - image file path
# returns - image matrix(in gray scale)
def image_load_gray(image_path):
    image_org = cv.imread(image_path)
    # cv.imshow('Original Image', image_org)
    # cv.waitKey(0)
    # convert the image into GRAY scale(for LBP)
    image_gray = cv.cvtColor(image_org, cv.COLOR_BGR2GRAY)
    # cv.imshow('Gray-scale Image', image_gray)
    # cv.waitKey(0)
    return image_gray


# Computes LBP of image window
# arguments - pixel block of gray-scale image
# returns - LBP matrix
def img_window_lbp_compute(image_gray_window):
    lbp_window = feature.local_binary_pattern(image_gray_window, LBP_NOP, LBP_RADIUS, LBP_METHOD)
    return lbp_window


# Computes histogram of LBP matrix(of image window)
# arguments - LBP window matrix
# returns - histogram
def lbp_window_histogram_compute(lbp_window_concatenated):
    (hist, hist_bin_edges) = np.histogram(lbp_window_concatenated, density = True, bins = HIST_BINS, range = HIST_RANGE)
    return hist, hist_bin_edges


# Displays given histogram
# arguments - histogram array, histogram bin edges, image ID
# TODO: Improve view
def histogram_display_store(hist_array, hist_bin_edges, image_id):
    plt.bar(hist_bin_edges[:-1], hist_array, width = hist_bin_edges[1] - hist_bin_edges[0])
    plt.xlim(min(hist_bin_edges), max(hist_bin_edges))
    plt.savefig(OUTPUT_PATH + image_id + OUTPUT_LBP_HISTOGRAM_FILENAME)
    # plt.show()


# Calculates euclidean distance b/w 2 lists
# arguments - list 1 & list 2
# returns - euclidean distance b/w the lists
def euclidean_dist(hist_1, hist_2):
    sum_ele = 0
    for x, y in zip(hist_1, hist_2):
        sum_ele += (x - y) ** 2
    # print('Distance - ', (sum_ele ** 0.5))
    distance = sum_ele ** 0.5
    return distance


# Calculates chi-squared distance b/w 2 lists
# arguments - list 1 & list 2
# returns - chi-2 distance b/w the lists
def chisq_dist(hist_1, hist_2, eps = 1e-10):
    # compute the chi-squared distance
    distance = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
                            for (a, b) in zip(hist_1, hist_2)])
    return distance


# Extracts LBP for given image and computes corresponding histogram
# Loads image -> Computes LBP/histogram for each 100X100 pixel window -> Concatenates result -> Stores result
# arguments - dataset folder path, image ID
# returns - histogram array, histogram bin edges, image file path
# TODO: store result in file/DB
def lbp_extract_image(dataset_path, image_id):
    # Check if LBP already computed for image
    if image_id in image_lbp_dict:
        print('Feature descriptor already exists for image ID - ', image_id)
        # Returns [0]-> histogram array, [1]-> histogram bin edges, [2]-> Image file path
        return image_lbp_dict[image_id][0], image_lbp_dict[image_id][1], image_lbp_dict[image_id][2]
    image_path = get_path_from_id(dataset_path, image_id)
    image_gray = image_load_gray(image_path)
    # image size
    image_height, image_width = image_gray.shape
    # print("Image size = {} X {} pixels".format(image_width, image_height))
    lbp_window_concatenated = []
    # Split image into 100X100 windows -> Compute LBP -> Concatenate vectors
    for y in range(0, image_height - 1, BLOCK_HEIGHT):
        for x in range(0, image_width - 1, BLOCK_WIDTH):
            image_gray_window = image_gray[y:y+BLOCK_HEIGHT, x:x+BLOCK_WIDTH]
            # print(image_gray_window)  # TEST
            lbp_window = img_window_lbp_compute(image_gray_window)
            # print(lbp_window)  # TEST
            # Concatenating LBP results(flattening LBP matrices)
            lbp_window_concatenated.append(lbp_window.ravel())
    # LBP Histogram(normalized; for LBP concatenated vector)
    hist, hist_bin_edges = lbp_window_histogram_compute(lbp_window_concatenated)
    # print('Histogram final vector - ', hist_concatenated)
    # print('Histogram bin edges - ', hist_bin_edges)
    # Storing result(image_id -> histogram array, hist_bin_edges, image file path)
    image_lbp_dict[image_id] = [hist, hist_bin_edges, image_path]
    return hist, hist_bin_edges, image_path


# Extracts LBP for every image in given folder -> Stores in image LBP feature dictionary
# arguments - folder path
def lbp_extract_folder(dataset_path):
    # Iterate over each image, find image ID and store LBP result in feature dictionary
    for image_filename in os.listdir(dataset_path):
        try:
            image_id = re.search('Hand_([0-9]+).jpg$', image_filename).group(1)
        except AttributeError:
            continue
        hist, hist_bin_edges, _ = lbp_extract_image(dataset_path, image_id)
        histogram_display_store(hist, hist_bin_edges, image_id)
    # Store LBP feature dictionary in file(CSV)
    output_absolute_path = OUTPUT_PATH + OUTPUT_LBP_FOLDER_FILENAME
    with open(output_absolute_path, 'w') as outfile:
        writer = csv.writer(outfile, delimiter = ',')
        writer.writerow(['image_id', 'histogram_array', 'histogram_bin_edges', 'image_filepath'])
        for k in sorted(image_lbp_dict.keys()):
            writer.writerow([k] + image_lbp_dict[k])
    print('Output saved for directory {} \n in file - \n{}'.format(dataset_path, output_absolute_path))


# LBP similarity measure to find 'k' most similar images to given image
# arguments - input folder, image ID, 'k' value
def lbp_similarity_measure(dataset_path, image_id, k_similar):
    hist, _, _ = lbp_extract_image(dataset_path, image_id)
    similarity_dict = {}
    # Find distance of given image from every image in feature dictionary
    for k, v in image_lbp_dict.items():
        if k != image_id:
            # similarity_dict[k] = [euclidean_dist(hist, v[0]), v[2]]  # Euclidean
            similarity_dict[k] = [chisq_dist(hist, v[0]), v[2]]  # Chi-squared
        else:
            continue
    # Sorting distances in increasing order
    similarity_sorted = sorted(similarity_dict.items(), key=lambda kv: kv[1])
    print("Details of 'k' most similar images - Image ID : Euclidean distance(similarity measure)")
    similarity_sorted = similarity_sorted[:k_similar]
    similar_images = [mpimg.imread(get_path_from_id(dataset_path, image_id))]
    similar_images_titles = [image_id]
    # Printing 'k' least distances OR 'k' most similar image IDs
    for ele in similarity_sorted:
        print('{} : {}'.format(ele[0], ele[1][0]))
        similar_images.append(mpimg.imread(ele[1][1]))
        similar_images_titles.append(ele[0])
    # Display similar images
    figure, axes = plt.subplots(1, len(similar_images))
    rank = 0
    for i, image in enumerate(similar_images):
        axes[i].imshow(image)
        axes[i].set_title('Rank: {}\nID: {}'.format(rank, similar_images_titles[i]))
        rank += 1
        axes[i].axis('off')
    plt.suptitle('Most similar k(={}) images[Given image - Rank = 0]'.format(k_similar))
    plt.show()


# Checks for user choice and performs corresponding Task:
# 1 - Task 1, 2 - Task 2, 3 - Task 3, 4 - Display feature dictionary, 9 - QUIT
def demo():
    while True:
        option = input(
            "\nEnter the task option to execute(or 9 to quit) \n"
            "1. Task 1 (extract and print feature descriptors for given image ID) \n"
            "2. Task 2 (extract and stores feature descriptors for all images in given folder) \n"
            "3. Task 3 (pre-requisite: Task 2; given an image ID- finds 'k' most similar images) \n"
            "4. Display the feature descriptor for a previously computed image ID from the current feature dictionary\n"
            "9. Quit\n"
            "Option? ")
        if option == '1':
            print("TASK 1:\n")
            # Sample - /home/varun/Desktop/MWDB/Project/input
            dataset_path = input('Enter the dataset folder path: ')
            # Check if directory exists
            if not os.path.isdir(dataset_path):
                print('Incorrect path - {}\n'
                      'Directory does not exist.'.format(dataset_path))
            # Sample - 0008110
            image_id = input('Enter the image id: ')
            image_path = get_path_from_id(dataset_path, image_id)
            # Check if image exists
            if path.exists(image_path) is False:
                print('Incorrect path - {}\n'
                      'Image does not exist.'.format(image_path))
                exit(1)
            # Extract LBP of image & Save in text file
            hist_array, hist_bin_edges, _ = lbp_extract_image(dataset_path, image_id)
            output_lbp_image = [image_id, hist_array, hist_bin_edges, image_path]
            output_absolute_path = OUTPUT_PATH + image_id + OUTPUT_LBP_IMAGE_FILENAME
            with open(output_absolute_path, 'w') as outfile:
                writer = csv.writer(outfile, delimiter = ',')
                writer.writerow(['image_id', 'histogram_array', 'histogram_bin_edges', 'image_filepath'])
                writer.writerow(output_lbp_image)
            print('Output for image ID - {} \nSaved in file - \n{}'.format(image_id, output_absolute_path))
            # Display histogram of LBP output & Save as Image
            np.set_printoptions(threshold = sys.maxsize)
            print('Histogram corresponding to image LBP feature descriptors - \n'
                  'Histogram array \n'
                  '{} \n'
                  'Histogram bin edges \n'
                  '{}'
                  .format(hist_array, hist_bin_edges))
            histogram_display_store(hist_array, hist_bin_edges, image_id)
        elif option == '2':
            print("TASK 2:")
            # Sample - /home/varun/Desktop/MWDB/Project/input
            dataset_path = input('Enter the dataset folder path: ')
            # Extract & Store LBP for each image in folder
            lbp_extract_folder(dataset_path)
        elif option == '3':
            print("TASK 3:")
            # Sample - /home/varun/Desktop/MWDB/Project/input
            dataset_path = input('Enter the dataset folder path: ')
            # Sample - 0008110
            image_id = input('Enter the image id: ')
            # Sample - 3
            k_similar = int(input("Enter value for 'k' most similar images: "))
            lbp_similarity_measure(dataset_path, image_id, k_similar)
        elif option == '4':
            # Sample - 0008110
            image_id = input('Enter the image ID: ')
            # Check if descriptor exists
            if image_id not in image_lbp_dict:
                print('Feature descriptors do not exist for image ID(not processed yet) - ', image_id)
                continue
            print('LBP FEATURE DICTIONARY for Image ID: ', image_id)
            print('Image ID: [Histogram array], [Histogram bin edges], Image file path')
            print(image_id + ' : \n' + str(image_lbp_dict[image_id][0]) + '\n'
                  + str(image_lbp_dict[image_id][1]) + '\n'
                  + str(image_lbp_dict[image_id][2]))
            histogram_display_store(image_lbp_dict[image_id][0], image_lbp_dict[image_id][1], image_id)
        elif option == '9':
            print("You opted to quit the demo. Exiting.")
            exit(0)
        else:
            print("Incorrect option. Please try again.")


# Run the demo cli
demo()
