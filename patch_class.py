# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 16:12:20 2020

@author: Daniel Sherman
"""

use_argv = True #False means I am training locally

import tensorflow
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
from skimage import color
from skimage.transform import resize
from skimage.filters import threshold_otsu
import pandas as pd
import glob
import slideio
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard
import imageio

#get to building the actual model

from keras import models

import tensorflow.keras.layers as L

from keras import backend as K

import sklearn
from keras.utils import to_categorical

import datetime

#https://elitedatascience.com/keras-tutorial-deep-learning-in-python

import os
os.system('cls' if os.name == 'nt' else 'clear')

plt.close('all')

initialize_start = time.time()

def show_image(image, title='Image', cmap_type='gray'):
    """
    Parameters
    ----------
    image : ndarray or image like file type
    title : str, optional
        The title of the plot. The default is 'Image'.
    cmap_type : str, optional
        Color map of the figure. The default is 'gray'.

    Returns
    -------
    None. Plots an image in greyscale
    """
    plot1 = plt.figure()
    plt.imshow(image, cmap=cmap_type)
    plt.title(title)
    plt.axis('on')
    plt.show()
    
def white_out(image):
    """
    Parameters
    ----------
    image : ndarray or image like file type

    Returns
    -------
    out : ndarray of the whited out image
    Selects all pixels almost equal to white (bit of fuzziness because there are edge effects in jpegs)
    """

    white = np.array([255, 255, 255])
    mask = np.abs(image - white).sum(axis=2) < 0.05

    # Find the bounding box of those pixels
    coords = np.array(np.nonzero(~mask))
    top_left = np.min(coords, axis=1)
    bottom_right = np.max(coords, axis=1)

    out = image[top_left[0]:bottom_right[0],
            top_left[1]:bottom_right[1]]
    return out

def clear_scale(image):
    """
    Parameters
    ----------
    image : ndarray of biopsy image

    Returns
    -------
    image : ndarray with scale bar replaced with white lines
    
    Function detects whether there are vertical or horizontal white bars.
    'Whites out' the scale bar depending on the direction of the white bars
    """
    
    if np.all(image[:, 5:7, :] == [255,255,255]):
        #check if an entire COLUMN is white
        image[1800:1850, 30:350, :] = [255, 255, 255]
    elif np.all(image[5:7, :, :] == [255,255,255]):
        #if an entire ROW is white
        image[1815:1855, 38:300, :] = [255,255,255]
    
    return image

def color_histo(img, title='Image'):
    """
    Parameters
    ----------
    img : ndarray of an RGB image
    title : str, optional
        Desired title. The default is 'Image'.

    Returns
    -------
    None.
    Plots figure 2x2 figure of the original image with the red, green, and blue histograms

    """
    reds = img[:,:,0]
    r_thresh = threshold_otsu(reds)
    
    green = img[:,:,1]
    g_thresh = threshold_otsu(green)
    
    blue = img[:,:,2]
    b_thresh = threshold_otsu(blue)
    
    fig = plt.figure()
    
    plt.subplot(2,2,1)
    plt.imshow(img)
    plt.title(title)
    
    plt.subplot(2,2,2)
    plt.hist(reds.flatten(order='C'), color='Red')
    plt.axvline(x=r_thresh, color='r', linestyle='dashed')
    plt.title('Red Histogram')
    plt.xlabel('Intensity')
    plt.ylabel('Pixel Count')
    
    plt.subplot(2,2,3)
    plt.hist(green.flatten(order='C'), color='Green')
    plt.axvline(x=g_thresh, color='g', linestyle='dashed')
    plt.title('Green Histogram')
    plt.xlabel('Intensity')
    plt.ylabel('Pixel Count')
    
    plt.subplot(2,2,4)
    plt.hist(blue.flatten(order='C'), color='Blue')
    plt.axvline(x=b_thresh, color='b', linestyle='dashed')
    plt.title('Blue Histogram')
    plt.xlabel('Intensity')
    plt.ylabel('Pixel Count')
    
    plt.show()
    return

def Sort(sub_li, i):
    sub_li.sort(key = lambda x:x[i], reverse=True)
    return sub_li

class Patch:
    """
    Class performs all preprocessing on image.
    Finds patches, subpatches, and stores them in a list
    """
    
    def __init__(self, img, label, sublabel, patNum = 3, patLen = 512, subPatNum = 5, subPatLen = 64):
        """
        Parameters
        ----------
        img : ndarray
            Image to find patches and subpatches in. MUST be run through function prep_image().
        label : str
            String containing diagnosis of the image (should be either 'Benign' or 'NonBenign').
        sublabel : str
            String containing the subclass of the image.
        patNum : int, optional
            Number of patches that the function will select. The default is 3.
        patLen : int, optional
            The length of the patch, please keep this as a power of 2. The default is 128.
        subPatNum : int, optional
            Number of subpatches that the function will select in a patch. The default is 5.
        subPatLen : int, optional
            The length of the smallest resolution subpatch, please keep this as a power of 2. The default is 32.

        Returns
        -------
        None.

        """
        self.wsi = img #input image
        self.label = label #OUTPUT image label
        self.sublabel = sublabel #OUTPUT image sublabel
        self.patNum = patNum
        self.patLen = patLen
        self.subPatNum = subPatNum #effectively 1 less
        self.subPatLen = subPatLen

        self.r256 = resize(img, (1024, 1024), anti_aliasing=True)
        self.r512 = resize(img, (2048, 2048), anti_aliasing=True)
        self.r1024 = resize(img, (4096, 4096), anti_aliasing=True)

        self.thresh = self.thresholding()[0]
        self.bin_mask = self.thresholding()[1]
       
        self.patLoc = self.find_pat()[3]
        self.patList256 = self.find_pat()[0]
        self.patList512 = self.find_pat()[1]
        self.patList1024 = self.find_pat()[2]
        
        self.subPatList256 = []
        self.subPatList512 = []
        self.subPatList1024 = []
        
        #creates list of ndarray of the subpatches
        for i in range(self.patNum):
            self.subPatList256.append(self.find_subPat(self.patList256[i], self.patList512[i], self.patList1024[i])[0]) 
            self.subPatList512.append(self.find_subPat(self.patList256[i], self.patList512[i], self.patList1024[i])[1]) 
            self.subPatList1024.append(self.find_subPat(self.patList256[i], self.patList512[i], self.patList1024[i])[2]) 
        
        self.subPat_lo = []
        self.subPat_mid = []
        self.subPat_hi = []
        
        for i in range(len(self.subPatList256)):
            for j in range(len(self.subPatList256[i])):
                self.subPat_lo.append(self.subPatList256[i][j]) 
                self.subPat_mid.append(self.subPatList512[i][j]) 
                self.subPat_hi.append(self.subPatList1024[i][j]) #OUTPUT
        
    def thresholding(self):
        """
        Returns
        -------
        self.thresh: float
            The binary threshold of pixel values that differentiates background/non-background.
        bin_mask : ndarray
            Mask of the pixels that are determined to be non-background.
        """
        
        self.thresh = threshold_otsu(self.r256)
        bin_mask = self.r256 < self.thresh
        bin_mask = bin_mask*1
        
        #print('Global Threshold is:', self.thresh)
        #show_image(bin_mask, title='256 Resolution Binary Mask')
        
        return self.thresh, bin_mask
        
    def ranCoord(self, image, s):
        """
        Parameters
        ----------
        image : ndarray
            Low resolution image.
        s : int
            Side length of patch.

        Returns
        -------
        list
            Coordinates of a random patch based on finding a random top-left corner location.
        """
     
        return [random.randint(0, image.shape[0]-s), random.randint(0, image.shape[1]-s)]
    
    def find_pat(self):
        """
        Divides an image into horizontal bands, then divides the bands to get boxes of particular lide length
        Orders the bands by decreasing content of 'non-background'

        Returns
        -------
        pat_list256 : ndarray
            List of the actual low res patch.
        pat_list512 : ndarray
            List of the actual mid res patch.
        pat_list1024 : ndarray
            List of the actual hi res patch.
        coord_list : list
            Coordinates of the top left corner in each of the patches on the low res image.
        """
        
        #utilizes the band and box method to hone in on a patch containing stroma
        img_len = self.bin_mask.shape[0]
    
        bands = []
        img_bands256 = []
        img_bands512 = []
        img_bands1024 = []
        
        # divides image into horizontal bands
        for i in range(int(img_len/self.patLen)):
            bands.append(self.bin_mask[i*self.patLen:(i+1)*self.patLen, :])
            img_bands256.append(self.r256[i*self.patLen:(i+1)*self.patLen, :])
            img_bands512.append(self.r512[2*i*self.patLen:2*(i+1)*self.patLen, :])
            img_bands1024.append(self.r1024[4*i*self.patLen:4*(i+1)*self.patLen, :])
            
        box = []
        
        # divides bands into horizontal boxes
        for i in range(int(img_len/self.patLen)):
            for j in range(int(img_len/self.patLen)):
                box.append([bands[i][:, j*self.patLen:(j+1)*self.patLen], 
                            np.count_nonzero(bands[i][:, j*self.patLen:(j+1)*self.patLen]==1), 
                            [i*self.patLen, j*self.patLen], 
                            img_bands256[i][:, j*self.patLen:(j+1)*self.patLen], 
                            img_bands512[i][:, 2*j*self.patLen:2*(j+1)*self.patLen], 
                            img_bands1024[i][:, 4*j*self.patLen:4*(j+1)*self.patLen]])
        
        #box contains [IMAGE, stroma_count, top_left_coord]
        patches = Sort(box, 1)
        
        pat_list256 = []
        pat_list512 = []
        pat_list1024 = []
        coord_list = []
        for i in range(self.patNum):
            pat_list256.append(patches[i][3])
            pat_list512.append(patches[i][4])
            pat_list1024.append(patches[i][5])
            coord_list.append(patches[i][2])
        
        return pat_list256, pat_list512, pat_list1024, coord_list
    
    def find_subPat(self, pat_lo, pat_mid, pat_hi):
        """
        Parameters
        ----------
        pat_lo : ndarray
            Low resolution patch.
        pat_mid : ndarray
            Mid resolution patch.
        pat_hi : ndarray
            Hi resolution patch.

        Returns
        -------
        subPatList256 : ndarray
            List of low resolution subpatches.
        subPatList512 : ndarray
            List of mid resolution subpatches.
        subPatList1024 : ndarray
            List of hi resolution subpatches.
        coord_list : list
            List of the top left coordinates of the subpatches (local coordinates on the patch).
        """
        
        #does what self.find_pat() does on the subpatch level, and therefore needs input of the patch at each resolution
        bin_mask = pat_lo < self.thresh
        bin_mask = bin_mask*1
        
        bands = []
        pat_bands256 = []
        pat_bands512 = []
        pat_bands1024 = []
        
        # divides image into horizontal bands
        for i in range(int(self.patLen/self.subPatLen)):
            bands.append(bin_mask[i*self.subPatLen:(i+1)*self.subPatLen, :])
            pat_bands256.append(pat_lo[i*self.subPatLen:(i+1)*self.subPatLen, :])
            pat_bands512.append(pat_mid[2*i*self.subPatLen:2*(i+1)*self.subPatLen, :])
            pat_bands1024.append(pat_hi[4*i*self.subPatLen:4*(i+1)*self.subPatLen, :])
            
        box = []
        
        # divides bands into boxes
        for i in range(int(self.patLen/self.subPatLen)):
            for j in range(int(self.patLen/self.subPatLen)):
                box.append([bands[i][:, j*self.subPatLen:(j+1)*self.subPatLen], 
                            np.count_nonzero(bands[i][:, j*self.subPatLen:(j+1)*self.subPatLen]==1),
                            pat_bands256[i][:, j*self.subPatLen:(j+1)*self.subPatLen],
                            pat_bands512[i][:, 2*j*self.subPatLen:2*(j+1)*self.subPatLen],
                            pat_bands1024[i][:, 4*j*self.subPatLen:4*(j+1)*self.subPatLen],
                            [i*self.subPatLen, (i+1)*self.subPatLen]
                    ])
        
        patches = Sort(box, 1)
        
        subPatList256 = []
        subPatList512 = []
        subPatList1024 = []
        coord_list = []
        
        for i in range(self.subPatNum):
            subPatList256.append(patches[i][2])
            subPatList512.append(patches[i][3])
            subPatList1024.append(patches[i][4])
            coord_list.append(patches[i][5])
        
        
        return subPatList256, subPatList512, subPatList1024, coord_list
    
    def makeRect(self, min_x, min_y, s):
        """
        Parameters
        ----------
        min_x : int
            Coordinate of the top left corner.
        min_y : int
            Coordinate of the top left corner.
        s : int
            Side length.

        Returns
        -------
        list
            4 Coordinates of a square.

        """
        #give top left point, returns top left and bottom right points
        min_x = min_x
        min_y = min_y
        max_x = min_x + s
        max_y = min_y + s
        
        return [min_x, min_y, max_x, max_y]
    
    def is_stroma(self, coord):
        """
        Parameters
        ----------
        coord : list
            Top left coordinate of patch.

        Returns
        -------
        bool
            Decision on whether or not the patch contains sufficient stroma.
        """
        
        #coord is the top left coordinate on the patch on the 256 resolution image
        s = self.patLen
        
        R1 = self.makeRect(coord[1], coord[0], s) 
        #for some reason, python takes the coord as [row, column] (which is [y,x])
        #instead of [x,y], but is fixed
        
        bin_patch = self.bin_mask[int(R1[0]):int(R1[2]), int(R1[1]):int(R1[3])]
        
        pat_mask = bin_patch > self.thresh
        pat_mask = pat_mask*1
        
        #show_image(bin_patch, title='bin_patch')
        #show_image(pat_mask, title='pat_mask')
        
        stroma_area = np.count_nonzero(pat_mask == 1)
        #print(stroma_area)
        pat_area = bin_patch.shape[0]*bin_patch.shape[1]
        #print(pat_area)
        
        stroma_ratio = stroma_area/pat_area
        print('[x,y]=[', coord[0], coord[1], ']', 'Stroma Ratio:', stroma_ratio)
        
        if(stroma_ratio >= 0.50):
            print('Stroma')
            return True
        else:
            print('Background')
            return False
        
    def is_subpat_stroma(self, patch, coord):
        """
        Parameters
        ----------
        patch : ndarray
            Patch that the subpatch is chosen from.
        coord : list
            Top left coordinate of the subpatch.

        Returns
        -------
        bool
            Decision on whether or not the subpatch contains enough stroma.
        """
        
        #for a given patch and top left coord of subpatch, determines if it contains stroma
        s = self.subPatLen
        
        R1 = self.makeRect(int(coord[1]), int(coord[0]), s)
        
        #for some reason, python takes the coord as [row, column] (which is [y,x])
        #instead of [x,y], but is fixed
        
        subpat = patch[int(R1[0]):int(R1[2]), int(R1[1]):int(R1[3])]
        subpat_mask = subpat < self.thresh
        subpat_mask = subpat_mask*1
        
        stroma_area = np.count_nonzero(subpat_mask == 1)
        subpat_area = subpat.shape[0]*subpat.shape[1]
        
        stroma_ratio = stroma_area/subpat_area
        
        print('Subpatch Coordinate: [x,y]=[', coord[0], coord[1], ']', 'Stroma Ratio:', stroma_ratio)
        
        if(stroma_ratio >= 0.4):
            print('Stoma')
            return True
        else:
            print('Background')
            return False
    
    def is_intersect(self, R1, R2):
        """
        Parameters
        ----------
        R1 : list
            List of ints representing the 4 corners of a rectangle.
        R2 : list
            List of ints representing the 4 corners of a rectangle.

        Returns
        -------
        bool
            Descision on whether or not the rectangles overlap.
        """
        
        # runs checks on the points from 2 instances of Rectangle
        # returns True if overlaps
        if R1[0] > R2[2] or R1[2] < R2[0]:
            return False
        if R1[1] > R2[3] or R1[3] < R2[1]:
            return False
        return True
       
    def getPat(self): 
        """
        LEGACY: Rolls N random patches of s side length and checks if any of them overlap, ensuring that they all contain sufficient stroma

        Returns
        -------
        coord_list : list
            List of ints containing the top left corner of the selected patches.

        """
        N = self.patNum
        s = self.patLen
        
        # get N s by s non-overlapping patches on the r256 image
        coord_list = -1*np.ones([N,2]) #dummy storage for patch coord
        
        rect_list = []
        for k in range(N):
            rect_list.append([-1, -1, -1, -1])
            
            
        str_chk = False
        while(not str_chk):
            coord_list[0,:] = self.ranCoord(self.r256, s)
            if(self.is_stroma(coord_list[0,:])):
                #print(colored('Patch 0 contains stroma', 'magenta'))
                str_chk = True
        
        #coord_list[0,:] = self.ranCoord() #roll first coordinate
        rect_list[0] = self.makeRect(coord_list[0,0], coord_list[0,1], s)
        
        count = 0
        isSolved = False

        '''begin loop hell'''
        
        while(isSolved == False):
            #for the current patch, i, ...
            
            for i in range(1,N):
                print(i)
                while(count != i):
                    
                    str_chk = False
                    while(not str_chk):
                        coord_list[i,:] = self.ranCoord(self.r256, s)
                        if(self.is_stroma(coord_list[i,:])):
                            print('Patch', i, 'containts stroma')
                            str_chk= True                    
                    
                    #coord_list[i,:] = self.ranCoord()
                    rect_list[i] = self.makeRect(coord_list[i,0], coord_list[i,1], s)
                    
                    #then, loop through previous patches to determine overlap
                    for j in range(i, -1, -1):             
                        if(not(self.is_intersect(rect_list[i], rect_list[j]))):
                            count +=1
                            print('Patch',i,'result is:')
                            print(i, 'Incremented count:', j, 'left')
                        else:
                            count = 0
                            if(self.is_intersect(rect_list[i], rect_list[j]) and i!=j):
                                print('Patch', i, 'and', j, 'overlap')
                                #continue

                                
            #print('Finished Collecting Patches')
            
            #calling is_stroma() for each patch
            for k in range(N):
                #print('Stroma Check Patch', k)
                self.is_stroma(coord_list[k,:])
            
            if(count == N-1):
                isSolved = True
            
            print('Patch Coordinates:')
            print(coord_list)
                       
        return coord_list
    
    def show_patch(self, image, xy, cmap_type='gray', title='image'):
        """
        Parameters
        ----------
        image : ndarray
            The patch image.
        xy : list
            Coordinates of patch.
        cmap_type : str, optional
            Color map of image. The default is 'gray'.
        title : str, optional
            Title of image. The default is 'image'.

        Returns
        -------
        bool
            Plots image of selected patch.
        """
        
        s = self.patLen
        my_extent = [xy[0], xy[0]+s, xy[1], xy[1]+s]
        #modified show_image function to show the patches and their locations with extent
        #function plots an image in greyscale
        #plot1 = plt.figure()
        plt.imshow(image, cmap=cmap_type, extent=my_extent)
        plt.title(title)
        plt.axis('on')
        plt.show()
        
        return True
    
    def patch_plot(self):
        """
        Returns
        -------
        bool
            Plots patches.

        """
        #plots patches marked on different resolution images by changing pizel values
        s = self.patLen
        safe_256 = self.r256
        pat_256 = np.zeros((len(self.patLoc), s, s))
        pat_512 = np.zeros((len(self.patLoc), 2*s, 2*s))
        pat_1024 = np.zeros((len(self.patLoc), 4*s, 4*s))

        for i in range(len(self.patLoc)):
            #265 patch collection
            pat_256[i,:,:] = self.r256[int(self.patLoc[i][1]):int(self.patLoc[i][1])+s, int(self.patLoc[i][0]):int(self.patLoc[i][0])+s]
            #show_image(pat_256[i,:,:], title='256 Patch')
        
            safe_256[int(self.patLoc[i][1]-1):int(self.patLoc[i][1]+1)+s, int(self.patLoc[i][0]-1)] = 0 #top x axis
            safe_256[int(self.patLoc[i][1]-1):int(self.patLoc[i][1]+1)+s, int(self.patLoc[i][0]+1)+s] = 0 #bottom x axis
            safe_256[int(self.patLoc[i][1]-1), int(self.patLoc[i][0]-1):int(self.patLoc[i][0]+1)+s] = 0 #left y axis
            safe_256[int(self.patLoc[i][1]+1)+s, int(self.patLoc[i][0]-1):int(self.patLoc[i][0]+1)+s] = 0 #right y axis
            
            #512 patch collection
            pat_512[i,:,:] = self.r512[2*(int(self.patLoc[i][1])):2*(int(self.patLoc[i][1]+s)), 2*(int(self.patLoc[i][0])):2*(int(self.patLoc[i][0]+s))]
            #show_image(pat_512[i,:,:], title='512 Patch')
            
            self.r512[2*int(self.patLoc[i][1]-1):2*(int(self.patLoc[i][1]+1)+s), 2*int(self.patLoc[i][0])-1] = 0 #top x axis
            self.r512[2*int(self.patLoc[i][1]-1):2*(int(self.patLoc[i][1]+1)+s), 2*(int(self.patLoc[i][0]+1)+s)] = 0 #bottom x axis
            self.r512[2*int(self.patLoc[i][1]-1), 2*(int(self.patLoc[i][0]-1)):2*(int(self.patLoc[i][0]+1)+s)] = 0 #left y axis
            self.r512[2*(int(self.patLoc[i][1]+1)+s), 2*int(self.patLoc[i][0]-1):2*(int(self.patLoc[i][0]+1)+s)] = 0 #right y axis
            
            #1024 patch collection
            pat_1024[i,:,:] = self.r1024[4*(int(self.patLoc[i][1])):4*(int(self.patLoc[i][1]+s)), 4*(int(self.patLoc[i][0])):4*(int(self.patLoc[i][0]+s))]
            #show_image(pat_512[i,:,:], title='512 Patch')
            
            self.r1024[4*int(self.patLoc[i][1]-1):4*(int(self.patLoc[i][1]+1)+s), 4*int(self.patLoc[i][0]-1)] = 0 #top x axis
            self.r1024[4*int(self.patLoc[i][1]-1):4*(int(self.patLoc[i][1]+1)+s), 4*(int(self.patLoc[i][0]+1)+s)] = 0 #bottom x axis
            self.r1024[4*int(self.patLoc[i][1]-1), 4*(int(self.patLoc[i][0]-1)):4*(int(self.patLoc[i][0]+1)+s)] = 0 #left y axis
            self.r1024[4*(int(self.patLoc[i][1]+1)+s), 4*int(self.patLoc[i][0]-1):4*(int(self.patLoc[i][0]+1)+s)] = 0 #right y axis
        
        show_image(safe_256, title='256px Resolution-Patches Marked')
        #show_image(self.r512, title='512 patches marked')
        #show_image(self.r1024, title='1024 patches marked')
        
        return True
    
    def store_patch(self):
        """
        Stores patch in a list
        
        Returns
        -------
        p_256 : ndarray
            Low resolution patch.
        p_512 : ndarray
            Mid resolution patch.
        p_1024 : ndarray
            Hi resolution patch.
        """
        
        #takes patch location and side length and stores the patch image information
        #for all three resolutions
        p_256 = []
        p_512 = []
        p_1024 = []
        s = self.patLen
        
        for i in range(len(self.patLoc)):
            p_256.append(self.r256[int(self.patLoc[i][1]):int(self.patLoc[i][1]+s), 
                                   int(self.patLoc[i][0]):int(self.patLoc[i][0]+s)])
            p_512.append(self.r512[2*int(self.patLoc[i][1]):2*int(self.patLoc[i][1]+s), 
                                   2*int(self.patLoc[i][0]):2*int(self.patLoc[i][0]+s)])
            p_1024.append(self.r1024[4*int(self.patLoc[i][1]):4*int(self.patLoc[i][1]+s), 
                                    4*int(self.patLoc[i][0]):4*int(self.patLoc[i][0]+s)])
        
        return p_256, p_512, p_1024
    
    def store_subpatch(self, patch):
        """
        Stores subpatch ndarrays in a list

        Parameters
        ----------
        patch : ndarray
            Patch that subpatches will be taken from.

        Returns
        -------
        sp_256 : list
            Subpatches.
        sp_1024 : list
            Subpatches.

        """
        #takes subpatch locations and side length and stores the subpatch image information
        #for all three resolutions
        s = self.subPatLen
        
        sp_256 = []
        sp_512 = []
        sp_1024 = []
        
        subPatLocs = self.find_subPat(patch)[1]
        
        for i in range(len(subPatLocs)):
            sp_256.append(patch[int(subPatLocs[i][1]):int(subPatLocs[i][1])+s, 
                                int(subPatLocs[i][0]):int(subPatLocs[i][0])+s])
            
            sp_512.append(patch[2*int(subPatLocs[i][1]):2*int(subPatLocs[i][1]+s), 2*int(subPatLocs[i][0]):2*int(subPatLocs[i][0]+s)])
            
        return sp_256, sp_512#, sp_1024
    
    def histo_show(self):
        """
        Plots histogram of the whole slide image

        Returns
        -------
        bool
            Dummy output.
        """
        
        plt.hist(self.wsi.ravel(), bins=256)
        plt.title('WSI Histogram')
        plt.show()
        print(self.wsi.ravel())
        print(np.average(self.wsi.ravel()))
        return True
    
    def getSubPat(self, patch): #this needs to be run for each patch
        """
        LEGACY: Chooses subpatches so they do not overlap and contain sufficient stroma
        
        patch: ndarray
            The actual patch image
        
        Returns
        -------
        coord_list: list
            Top left coordinates of the selected subpatches
        """
        #method takes a patch from self.store_patch() and runs all the same processing that self.getPat() does
        n = self.subPatNum
        s = self.subPatLen
        
        coord_list = -1*np.ones([n,2])
        
        rect_list = []
        for k in range(n):
            rect_list.append([-1,-1,-1,-1])
            
        str_chk = False
        while(not str_chk):
            coord_list[0,:] = self.ranCoord(patch, s)
            if(self.is_subpat_stroma(patch, coord_list[0,:])):
                print('Subpatch 0 contains stroma')
                str_chk = True
        
        rect_list[0] = self.makeRect(coord_list[0,0], coord_list[0,1], s)
        
        count = 0
        isSolved = False
        
        '''loop hell'''
        
        while(isSolved == False):
            #for the current subpatch i...
            for i in range(1,n):
                print(i) #subpatch indicator
                while(count != i):
                    
                    str_chk = False
                    while(not str_chk):
                        coord_list[i,:] = self.ranCoord(patch, s)
                        if(self.is_subpat_stroma(patch, coord_list[i,:])):
                            print('Subpatch', i, 'contains stroma')
                            str_chk = True
                    rect_list[i] = self.makeRect(coord_list[i,0], coord_list[i,1], s)
                    
                    #then loop through previous subpatches to determine overlap
                    for j in range(i, -1, -1):
                        if(not(self.is_intersect(rect_list[i], rect_list[j]))):
                            count +=1
                            print('Subpatch', i, 'result is:')
                            print(i, 'Incremented count:', j, 'left')
                        else:
                            count = 0
                            if(self.is_intersect(rect_list[i], rect_list[j]) and i != j):
                                print('Subpatch', i, 'and', j, 'overlap')
                                continue
            
            print('Finished Collecting Subpatches')
            
            #calling is_subpat_stroma() on each subpatch
            for k in range(n):
                print('Stroma Check Subpatch', k)
                self.is_subpat_stroma(patch, coord_list[k, :])
            
            if(count == n-1):
                isSolved = True
            
            #print('Subpatch Coordinates:')
            #print(coord_list)
        
        return coord_list
    
def phantom_test():
    """
    Tests the is_stroma method on a phantom of a binary mask and checks output

    Returns
    -------
    None.

    """
    
    #TESTING THE BINARY MASK PHANTOM
    print('PHANTOM TESTING')
    #phantom_test = Patch(phantom)
    print('SHOULD BE:')
    print('Stroma')
    phantom_test.is_stroma([0,0],30)
    print('SHOULD BE:')
    print('Background')
    phantom_test.is_stroma([30,0], 30)
    
def prep_image(image):
    noScaleBar = clear_scale(image)
    crop = white_out(noScaleBar)
    bw_image = color.rgb2gray(crop)
    
    return bw_image

def horizontal_flip(image_array):
    #flips pixels on x axis
    return image_array[:, ::-1]

def vertical_flip(image_array):
    #flips pixels on y axis
    return image_array[::-1, :]

def image_augment(image_array):
    '''
    Parameters
    ----------
    image_array : ndarray
        The image that you want augmented.

    Returns
    -------
    None.

    '''
    hori = horizontal_flip(image_array)
    verti = vertical_flip(image_array)
    both = horizontal_flip(vertical_flip(image_array))
    return image_array, hori, verti, both

print('----- %s seconds to initialize Patch class -----' % (time.time() - initialize_start))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

'''IMPORT IMAGES'''

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import sys

if use_argv:
    path_root = sys.argv[1]
    svs_path = path_root + '/*.svs'
    df_path = path_root + '/Data_Collection_Labels_Sublabels.xlsx'
    print(glob.glob(sys.argv[1]))
    print(os.listdir(sys.argv[1]))
else:
    path_root = '//192.168.1.10/Data/HDD_Mt_Sinai'
    svs_path = path_root + '/IMAGES/BENIGN/*.svs'
    df_path = path_root + '/Data_Collection_Labels_Sublabels.xlsx'

import_time = time.time()
#import all images
file_path = []
for filename in glob.glob(svs_path): #assuming SVS
    file_path.append(filename)
print('----- %s seconds to import image paths -----' % (time.time() - import_time))
#print(len(file_path))
    
label_manage_start = time.time()

#change forward slash to backslash
for i in range(len(file_path)):
    file_path[i] = file_path[i].replace('\\', '/')


#take only image number to cross reference on the excel sheet of labels
file_name = []
for i in range(len(file_path)):
    #file_name.append(file_path[i][47:-4]) #VERY SPECIFIC TO THE LENGTH OF THE PATH
    file_name.append(file_path[i].replace(svs_path[0:-5], ''))
    file_name[i] = file_name[i].replace('.svs', '')
    
unique_id = ['1', '15', '4', '2'] #hand selected 1 image of each benign subclassification
unique_path = []

for id_ in unique_id:
    unique_path.append(svs_path[:-5]+ id_+'.svs')

#make a dataframe of the image names, file paths, and images themselves
if use_argv:
    d = {'file_name':file_name, 'file_path':file_path}
else:
    d = {'file_name':unique_id, 'file_path':unique_path}
    
image_info = pd.DataFrame(data=d)

#import excel spreadsheet

df = pd.read_excel(df_path, engine='openpyxl')
df = df.drop(['SP'], axis=1)
df = df.dropna(axis='index')

df['Participant ID'] = df['Participant ID'].astype(int)
df['Participant ID'] = df['Participant ID'].astype(str)

common_items = set(df['Participant ID']) & set(image_info['file_name'])
df_common = df[df['Participant ID'].isin(common_items)]
image_info_common = image_info[image_info['file_name'].isin(common_items)]
    
df_common = df_common.sort_values(by=['Participant ID'])
df_common = df_common.reset_index(drop=True)
image_info_common = image_info_common.sort_values(by=['file_name'])
image_info_common = image_info_common.reset_index(drop=True)

common_paths = []
for i in range(len(image_info_common)):
    common_paths.append(image_info_common['file_path'][i])
    
df_common['paths'] = common_paths

if 'Benign' and 'NonBenign' in df_common.Label.unique(): #df_common contains benign and nonbenign images
    #need to split image_info_common into 2 dataframes for benign and nonbenign images
    df_c_nonBenign, df_c_benign = [x for _, x in df_common.groupby(df_common['Label'] == 'Benign')]
    df_c_nonBenign = df_c_nonBenign.reset_index(drop=True)
    df_c_benign = df_c_benign.reset_index(drop=True)

    print('benign info:')
    print(df_c_benign.info(verbose=True))
    print('nonbenign info:')
    print(df_c_nonBenign.info(verbose=True))
    
elif 'Benign' in df_common.Label.unique() and not('NonBenign' in df_common.Label.unique()): #only contains benign images
    df_c_benign = df_common
    print('benign info:')
    print(df_c_benign.info(verbose=True))
    
elif 'NonBenign' in df_common.Label.unique() and not('Benign' in df_common.Label.unique()): #only contains nonbenign images
    df_c_nonBenign = df_common
    print('nonbenign info:')
    print(df_c_nonBenign.info(verbose=True))


#create list of numpy array images by using io.imread()
image_list = []

num_image = len(df_common['paths'])

for i in range(num_image): #normally loops len(file_path)
    img_load = time.time()
    print('Starting to load image '+str(i))
    slide = slideio.open_slide(df_common['paths'][i], 'SVS')
    scene = slide.get_scene(0)
    image = scene.read_block(size=(4096, 0)) #importing @ highest resolution
    image_list.append(image)
    print('Loaded image ' + str(i) + ' in ' + str(time.time() - img_load) + ' seconds')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

'''PHANTOM TESTING'''
#UNCOMMENT FOR DEBUGGING PATCH PROCESSING

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
#phantom = io.imread('phantom.png')
#show_image(phantom, 'Binary Mask Phantom')
#phantom = color.rgb2gray(phantom)

#phantom_test()

#pat_phantom = io.imread('patch_phantom.png')
#pat_phantom = color.rgb2gray(pat_phantom)

#pat_test = Patch(pat_phantom, 'test', 'test')
#pat_test.is_subpat_stroma(pat_phantom, [10,0], 5)
#pat_test.getSubPat(pat_phantom, 5, 5)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

'''IMPLEMENT ON ENDOMETRIAL SLIDE'''

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Patches = []
Labels = []

Benign_Patches = []
Benign_Sublabels = []

NonBenign_Patches = []
NonBenign_Sublabels = []

for img_num in range(len(image_list)):
    print('Processing', img_num+1, 'of', len(image_list))
    temp = Patch(prep_image(image_list[img_num]), df_common['Label'][img_num], df_common['Sublabel'][img_num])
    
    for j in range(len(temp.subPat_hi)):
        #append each augmented image, and the label for each one        
        for augmented in range(len(image_augment(temp.subPat_hi[j]))):
            Patches.append(image_augment(temp.subPat_hi[j])[augmented])
            Labels.append(temp.label)
        
        if temp.label == 'Benign':
            for augmented in range(len(image_augment(temp.subPat_hi[j]))):
                Benign_Patches.append(image_augment(temp.subPat_hi[j])[augmented])
                Benign_Sublabels.append(temp.sublabel)
                
        elif temp.label == 'NonBenign':
            for augmented in range(len(image_augment(temp.subPat_hi[j]))):
                NonBenign_Patches.append(image_augment(temp.subPat_hi[j])[augmented])
                NonBenign_Sublabels.append(temp.sublabel)
  
#there will always be at least 1 patch
Patches = np.asarray(Patches, dtype='float32')
Patches = np.reshape(Patches, (Patches.shape[0], Patches.shape[1], Patches.shape[2], 1))
print('Patches.shape', Patches.shape)
print('len(Labels):',len(Labels))

if np.size(Benign_Patches) > 0:
    Benign_Patches = np.asarray(Benign_Patches, dtype='float32')
    Benign_Patches = np.reshape(Benign_Patches, (Benign_Patches.shape[0], Benign_Patches.shape[1], Benign_Patches.shape[2], 1))
    print('Benign_Patches.shape', Benign_Patches.shape)
    print('len(Benign_Sublabels):',len(Benign_Sublabels))

if np.size(NonBenign_Patches) > 0:
    NonBenign_Patches = np.asarray(NonBenign_Patches, dtype='float32')
    NonBenign_Patches = np.reshape(NonBenign_Patches, (NonBenign_Patches.shape[0], NonBenign_Patches.shape[1], NonBenign_Patches.shape[2], 1))
    print('NonBenign_Patches.shape', NonBenign_Patches.shape)
    print('len(NonBenign_Sublabels):',len(NonBenign_Sublabels))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

'''ONE HOT ENCODE LABELS'''

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Label_le = sklearn.preprocessing.LabelEncoder()
Label_le.fit(Labels)
print(list(Label_le.classes_))
Label_OH = Label_le.transform(Labels) 
Label_OH = to_categorical(Label_OH)
print('Label_OH.shape:', Label_OH.shape)
#print('Label_OH:', Label_OH)
Patches_train, Patches_test, Label_train, Label_test = train_test_split(
    Patches, Label_OH, shuffle=True, random_state=42)

if np.size(Benign_Patches) > 0:
    Benign_Sublabels_le = sklearn.preprocessing.LabelEncoder() 
    Benign_Sublabels_le.fit(Benign_Sublabels) 
    print(list(Benign_Sublabels_le.classes_)) 
    Benign_Sublabels_OH = Benign_Sublabels_le.transform(Benign_Sublabels) 
    #print(Benign_Sublabels_OH) 
    Benign_Sublabels_OH = to_categorical(Benign_Sublabels_OH)
    print('Benign_Sublabels_OH.shape:', Benign_Sublabels_OH.shape)
    #print('Benign_Sublabels_OH',Benign_Sublabels_OH)
    B_Patches_train, B_Patches_test, B_Sublabels_train, B_Sublabels_test = train_test_split(
        Benign_Patches, Benign_Sublabels_OH, shuffle=True, random_state=42)

if np.size(NonBenign_Patches) > 0:
    NonBenign_Sublabels_le = sklearn.preprocessing.LabelEncoder()
    NonBenign_Sublabels_le.fit(NonBenign_Sublabels)
    print(list(NonBenign_Sublabels_le.classes_))
    NonBenign_Sublabels_OH = NonBenign_Sublabels_le.transform(NonBenign_Sublabels)
    NonBenign_Sublabels_OH = to_categorical(NonBenign_Sublabels_OH)
    print('NonBenign_Sublabels_OH.shape:', NonBenign_Sublabels_OH.shape)
    #print('NonBenign_Sublabels_OH:', NonBenign_Sublabels_OH)
    NB_Patches_train, NB_Patches_test, NB_Sublabels_train, NB_Sublabels_test = train_test_split(
        NonBenign_Patches, NonBenign_Sublabels_OH, shuffle=True, random_state=42)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

'''BUILD THE NETWORK'''

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

tensorflow.keras.backend.clear_session()

from myModel import UNet

tensorflow.keras.backend.clear_session()

#model=tensorflow.keras.Model(inputs=input_layer, outputs=outputs) #makes model out of variables
#model = AlexNet(input_layer, B_Sublabels_train)
#model = shallowNet(input_=B_Patches_train, output_=B_Sublabels_train)
model = UNet(B_Patches_train, B_Sublabels_train)

NUM_IMG = Patches.shape[0]
WIDTH = Patches.shape[1]
HEIGHT = Patches.shape[2]
    
input_layer = L.Input(shape=(WIDTH, HEIGHT, 1))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy', tf.keras.metrics.Precision(), 
                       tf.keras.metrics.Recall(), tf.keras.metrics.FalseNegatives(), 
                       tf.keras.metrics.FalsePositives()])

checkpoint_path = 'unet-view\\cp.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)

NAME = 'BENIGN-unet-big-patch-{}'.format(datetime.datetime.now().strftime("%Y_%m_%d-%H,%M,%S"))
tboard = TensorBoard(log_dir='logs/fit/{}'.format(NAME), profile_batch=0)

checkpointer = tensorflow.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                          monitor='val_acc',
                                                          verbose=1,
                                                          save_weights_only=True)

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

model.summary()

history = model.fit(x=B_Patches_train, 
          y=B_Sublabels_train,
          epochs=10,
          callbacks=[checkpointer, tboard])#, early_stop]) #actually trains the model with data

os.listdir(checkpoint_dir)

#print('EVALUATING NOW')
#print('B_Patches_test.shape:', B_Patches_test.shape)
#print('B_Sublabels_test.shape:', B_Sublabels_test.shape)

model.evaluate(x=B_Patches_test,
               y=B_Sublabels_test)

#TRYING TO VISUALIZE THE LAYERS
#https://towardsdatascience.com/visualizing-intermediate-activation-in-convolutional-neural-networks-with-keras-260b36d60d0


layer_outputs = [layer.output for layer in model.layers]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

activations = activation_model.predict(B_Patches_test)

first_layer_activation = activations[0]
print(first_layer_activation.shape)

#activations=activation_model.predict(B_Patches_train)
#print(activations.shape)

##############################################################################################################################
#PLOT ACTIVATIONS OF LAYERS
##############################################################################################################################
'''
layer_names = []
for layer in model.layers:
    layer_names.append(layer.name)
    
for layer_name, conv_activations in zip(layer_names, activations):
    print(layer_name)
    print(conv_activations.shape)

    
    #plots convolutional layer figures (make this into a gif)
    #https://towardsdatascience.com/basics-of-gifs-with-pythons-matplotlib-54dd544b6f30
    if len(conv_activations.shape) == 4:
        conv_activations -= conv_activations.mean()
        conv_activations /= conv_activations.std()
        
        filenames = []
        for b in range(conv_activations.shape[-1]):
            filename = f'{b}.png'
            filenames.append(filename)
            
            fig, axs = plt.subplots(6,10, facecolor='w', edgecolor='k')
            fig.suptitle(layer_name+'[i,:,:,'+str(b)+']' + ' feature maps')
            for i in range(axs.shape[0]):
                for j in range(axs.shape[1]):
                    axs[i,j].set_title(str(i*10+j))
                    axs[i,j].matshow(conv_activations[i*10+j,:,:,b])
                    axs[i,j].get_xaxis().set_visible(False)
                    axs[i,j].get_yaxis().set_visible(False)
            plt.savefig(filename)
            plt.close()
            
    with imageio.get_writer(layer_name+'.gif', mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
            
    for filename in filenames:
        os.remove(filename)
        '''