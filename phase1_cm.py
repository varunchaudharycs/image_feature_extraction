"""
Created on Sat Sep 14 10:39:48 2019
@author: varun
TASKS-
1. Color moments feature extraction for given image ID.
2. Color moments feature extraction for all images in given folder path.
3. Finding 'k' most similar images to given image ID based on color moments feature descriptors.
"""

import os
from os import path
from sys import exit
import numpy as np
import cv2 as cv
from skimage import feature
import matplotlib.pyplot as plt
import re
import csv
import time
import sys
import math
import ast
import scipy.stats
import statistics
import glob
import matplotlib.image as mpimg


# dictionary to store feature descriptors
# format = {image_id -> [color moments, image file path]}
image_cm_dict = {}
# image filename = IMAGE_PREFIX + image_id + IMAGE_SUFFIX
IMAGE_PREFIX = '/Hand_'
IMAGE_SUFFIX = '.jpg'
# Block size parameters
BLOCK_HEIGHT = 100
BLOCK_WIDTH = 100
# Output parameters
# Sample - /home/varun/Desktop/MWDB/Project/output/
OUTPUT_PATH = str(input('Enter the output folder to store all results: ') + '/')
OUTPUT_CM_FOLDER_FILENAME = 'cm_folder_' + time.strftime('%m%d%Y') + '_' + time.strftime('%H%M%S') + '.csv'
OUTPUT_CM_IMAGE_FILENAME = '_cm_image_' + time.strftime('%m%d%Y') + '_' + time.strftime('%H%M%S') + '.csv'


# Returns image file path- FOLDER path + PREFIX + ID + SUFFIX
# arguments - folder path, image ID
def get_path_from_id(dataset_path, image_id):
    return str(dataset_path + IMAGE_PREFIX + image_id + IMAGE_SUFFIX)


# loads original image -> converts to YUV for color moments
# arguments - image file path
# returns - image matrix(in YUV scale)
def image_load_yuv(image_path):
    image_org = cv.imread(image_path)
    # cv.imshow('Original Image', image_org)
    # cv.waitKey(0)
    # convert the image into YUV scale(for LBP)
    image_yuv = cv.cvtColor(image_org, cv.COLOR_BGR2YUV)
    # cv.imshow('YUV-scale Image', image_yuv)
    # cv.waitKey(0)
    return image_yuv


# Display and store color moments of an image
# arguments - input folder path, image ID and color moment vector of image
# displays
def cm_display_store(image_cm_dict, output_absolute_path):
    with open(output_absolute_path, 'w') as outfile:
        writer = csv.writer(outfile, delimiter = ',')
        writer.writerow(['image_id', 'color_moments'])
        for k in sorted(image_cm_dict.keys()):
            writer.writerow([k] + image_cm_dict[k])
    print('Output saved in file - \n', output_absolute_path)


# Calculates euclidean distance b/w 2 lists
# arguments - list 1 & list 2
# returns - euclidean distance b/w the lists
def euclidean_dist(list_1, list_2):
    sum_ele = 0
    for x, y in zip(list_1, list_2):
        sum_ele += (x - y) ** 2
    # print('Distance - ', (sum_ele ** 0.5))
    distance = sum_ele ** 0.5
    return distance


# Computes color moments for given image ID
# arguments - dataset folder path, image ID
# returns - color moment vector for entire image
def cm_extract_image(dataset_path, image_id):
    # Check if CM already computed for image
    if image_id in image_cm_dict:
        print('Feature descriptor already exists for image ID - ', image_id)
        # Returns color moments vector
        return image_cm_dict[image_id]
    image_path = get_path_from_id(dataset_path, image_id)
    image_yuv = image_load_yuv(image_path)
    # image size
    image_height, image_width, image_channels = image_yuv.shape
    # print("Image size = {} X {} pixels".format(image_width, image_height))
    cm_window_concatenated = []
    # Split image into 100X100 windows -> Compute color moments -> Concatenate vectors
    for y in range(0, image_height - 1, BLOCK_HEIGHT):
        for x in range(0, image_width - 1, BLOCK_WIDTH):
            image_yuv_window = image_yuv[y:y + BLOCK_HEIGHT, x:x + BLOCK_WIDTH]
            # print(image_yuv_window)  # TEST
            (window_mean, window_std) = cv.meanStdDev(image_yuv_window)
            # print('Window mean - {},\n Window std - {}'.format(window_mean, window_std))
            # Unpacking window mean
            window_mean = list(window_mean.flatten())
            window_mean.reverse()
            # Unpacking window std
            window_std = list(window_std.flatten())
            window_std.reverse()
            # Computing window skew
            skew = scipy.stats.skew(image_yuv_window)
            skew_y = statistics.mean(skew[:, 0])
            skew_u = statistics.mean(skew[:, 1])
            skew_v = statistics.mean(skew[:, 2])
            window_skewness = [skew_y, skew_u, skew_v]
            # Single color moment feature vector for window
            cm_window_concatenated.append(window_mean + window_std + window_skewness)
    # print('Color Moments for image Id - {}: \n{}'.format(image_id, cm_window_concatenated))
    # store result in feature dictionary
    image_cm_dict[image_id] = cm_window_concatenated
    return cm_window_concatenated


# Extracts color moments for every image in given folder -> Stores in image cm feature dictionary -> Saves output to CSV
# arguments - folder path
def cm_extract_folder(dataset_path):
    # Iterate over each image, find image ID and store LBP result in feature dictionary
    for image_filename in os.listdir(dataset_path):
        try:
            image_id = re.search('Hand_([0-9]+).jpg$', image_filename).group(1)
        except AttributeError:
            continue
        image_cm_dict[image_id] = cm_extract_image(dataset_path, image_id)
    # Store CM feature dictionary in file(CSV)
    output_absolute_path = OUTPUT_PATH + OUTPUT_CM_FOLDER_FILENAME
    cm_display_store(image_cm_dict, output_absolute_path)


# CM similarity measure to find 'k' most similar images to given image
# arguments - input folder, image ID, 'k' value
def cm_similarity_measure(dataset_path, image_id, k_similar):
    image_cm_dict[image_id] = cm_extract_image(dataset_path, image_id)
    image_path = get_path_from_id(dataset_path, image_id)
    image_yuv = image_load_yuv(image_path)
    # image size
    image_height, image_width, image_channels = image_yuv.shape
    total_descriptors = (image_height * image_width)/(BLOCK_HEIGHT * BLOCK_WIDTH)
    similarity_dict = {}
    # Find distance of given image from every image in feature dictionary
    for k, v in image_cm_dict.items():
        euclidean_dist_sum = 0
        if k != image_id:
            # Comparing color moment vector window by window
            for cm_window_a, cm_window_b in zip(image_cm_dict[image_id], image_cm_dict[k]):
                # Considering only Y channel
                cm_window_a_Y = [cm_window_a[0], cm_window_a[3], cm_window_a[6]]
                cm_window_b_Y = [cm_window_b[0], cm_window_b[3], cm_window_b[6]]
                euclidean_dist_sum += euclidean_dist(cm_window_a_Y, cm_window_b_Y)
            # Computing average of euclidean distance
            similarity_dict[k] = euclidean_dist_sum/total_descriptors
    # Sorting distances in increasing order
    similarity_sorted = sorted(similarity_dict.items(), key = lambda kv: kv[1])
    print("Details of 'k' most similar images - Image ID : Average euclidean distance(similarity measure)")
    similarity_sorted = similarity_sorted[:k_similar]
    similar_images = [mpimg.imread(get_path_from_id(dataset_path, image_id))]
    similar_images_titles = [image_id]
    # Printing 'k' least distances OR 'k' most similar image IDs
    for ele in similarity_sorted:
        print('{} : {}'.format(ele[0], ele[1]))
        similar_images.append(mpimg.imread(get_path_from_id(dataset_path, ele[0])))
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
            image_cm_dict_local = {image_id: cm_extract_image(dataset_path, image_id)}
            # Parameters - input folder, image ID, Color moment image vector, Output file path
            cm_display_store(image_cm_dict, OUTPUT_PATH + image_id + OUTPUT_CM_IMAGE_FILENAME)
            print('Color moments of image Id - {}: \n{}'.format(image_id, image_cm_dict[image_id]))
        elif option == '2':
            print("TASK 2:")
            # Sample - /home/varun/Desktop/MWDB/Project/input
            dataset_path = input('Enter the dataset folder path: ')
            # Extract & Store color moments for each image in folder
            # Extract & Store LBP for each image in folder
            cm_extract_folder(dataset_path)
            print('Output for directory {} - STORED'.format(dataset_path))
        elif option == '3':
            print("TASK 3:")
            # Sample - /home/varun/Desktop/MWDB/Project/input
            dataset_path = input('Enter the dataset folder path: ')
            # Sample - 0008110
            image_id = input('Enter the image id: ')
            # Sample - 3
            k_similar = int(input("Enter value for 'k' most similar images: "))
            cm_similarity_measure(dataset_path, image_id, k_similar)
        elif option == '9':
            print("You opted to quit the demo. Exiting.")
            exit(0)
        else:
            print("Incorrect option. Please try again.")


# Run the demo cli
demo()