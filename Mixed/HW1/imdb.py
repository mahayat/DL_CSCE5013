#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 16:47:45 2019

@author: bjlowe
"""
import numpy as np

#Create N training pairs (X,Y) that sample the circle of radius R at center C
class DataSet:
    def __init__(self, radius, center):
        self.R = radius
        self.C = center
    
    
    def generatePointInside(self):
        (x1, x2) = np.random.uniform(low = -1*self.R, high = self.R, size=(2,1))
        return np.array([[x1 + self.C[0]], [x2 + self.C[1]]], dtype=float)
    
    
    def generatePointOutside(self):
        (x1, x2) = np.random.uniform(low = -20, high = 20, size=(2,1))
        while (x1 >= -1*self.R + self.C[0] and x1 <= self.R + self.C[0] and x2 >= -1*self.R + self.C[0] and x2 <= self.R + self.C[0]):
            (x1, x2) = np.random.uniform(low = -20, high = 20, size=(2,1))       
        return np.array([[x1], [x2]], dtype=float)
    
    
    def createDataset(self, posSamples, negSamples):
        #Create number of positives
        self.X = self.generatePointInside()
        self.Y = np.array([[1]])
        for i in range(posSamples - 1):
            self.X = np.hstack((self.X, self.generatePointInside()))
            self.Y = np.hstack((self.Y, np.array([[1]])))
            
        #Create a number of negatives
        for i in range(negSamples):
            self.X = np.hstack((self.X, self.generatePointOutside()))
            self.Y = np.hstack((self.Y, np.array([[0]])))
            
        #Shuffle them together randomly...
        #Get a permutation of the indices
        permutation = np.arange(posSamples+negSamples)
        np.random.shuffle(permutation)
        #Initialize the shuffled arrays by doing first action...
        self.shuffX = self.X[:,permutation[0]]
        self.shuffY = self.Y[:,permutation[0]]
        #remove first index
        permutation = np.delete(permutation, 0)
        #do the rest by stacking horizontally (adding columns)
        for i in permutation:
            self.shuffX = np.hstack((self.shuffX,self.X[:, i]))
            self.shuffY = np.hstack((self.shuffY,self.Y[:, i]))

        #reshape the Y to have different samples as columns
        self.shuffY = self.shuffY.reshape(1, -1)
        return (self.shuffX, self.shuffY)