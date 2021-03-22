# -*- coding: utf-8 -*-
"""Color-base.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1pd4_d_-Ku9Wg4s4OciLbmV2H8yvqnIcg
"""

import sys
import os
import numpy as np
import cv2
import math
import ntpath




def startCount(filePath, fileName):
  img = cv2.imread(filePath)
  hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

  # Thresh hold
  gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  blured = cv2.GaussianBlur(gray1,(5,5),0)
  ret, thresh = cv2.threshold(blured,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
  # cv2_imshow(thresh)

  # Range for lower red
  lower_red = np.array([0,70,0])
  upper_red = np.array([40,255,255])

  lower_white = np.array([0,70,0])
  upper_white = np.array([30,255,255])

  mask1 = cv2.inRange(hsv_img, lower_white, upper_white)
  # Range for upper range
  lower_red = np.array([170,70,0])
  upper_red = np.array([180,255,255])
  mask2 = cv2.inRange(hsv_img,lower_red,upper_red)

  mask = mask1+mask2

  result = cv2.bitwise_and(img.copy(), img.copy(), mask=mask)
  # cv2_imshow(result)

  # cv2_imshow(RGB_again)

  # Thresh hold

  red = result.copy()
  # set blue and green channels to 0
  red[:, :, 0] = 0
  red[:, :, 1] = 0

  gray = cv2.cvtColor(red, cv2.COLOR_BGR2GRAY)
  blured = cv2.GaussianBlur(gray,(5,5),0)
  ret, thresh = cv2.threshold(blured,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
  # cv2_imshow(gray)
  # cv2_imshow(thresh)

  kernel = np.ones((5,5),np.uint8)
  big = thresh.copy()
  closing = cv2.morphologyEx(big, cv2.MORPH_CLOSE, kernel)
  # cv2_imshow(closing)



  #Find max area
  negative = cv2.bitwise_not(closing)
  #Find contour
  contours, hierarchy = cv2.findContours(negative, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

  print('First contour', len(contours))

  hierarchy = hierarchy[0]
  max_area = cv2.contourArea(contours[0])
  total = 0
  for con in contours:
    area = cv2.contourArea(con)
    total += area
    if area > max_area:
      max_area = area


  diff = 0.1
  diff_avg = 0.7
  max_area = int(max_area * diff)
  average = int(total / (len(contours)))
  radius_avg = int(math.sqrt(average / 3.14))
  average = int(average * diff)
  print("Max size:", max_area)
  print("Average:", average)
  print("Average Radius:", radius_avg)

  # Remove small object
  mask = np.zeros(closing.shape[:2],dtype=np.uint8)

  #  For each contour, find the bounding rectangle and draw it
  for component in zip(contours, hierarchy):
      currentContour = component[0]
      currentHierarchy = component[1]
      area = cv2.contourArea(currentContour)
      # color_mask = np.zeros(mask.shape, np.uint8)
      # cv2.drawContours(color_mask, currentContour, -1, 255, -1)
      # mean = cv2.mean(color_mask, mask=color_mask)
      # print(mean)
      if currentHierarchy[3] < 0:
        if area > average:
          #  If contour inside, delete it
          cv2.drawContours(mask, [currentContour], 0, (255), -1)  
          
  # cv2_imshow(mask)

  #Find contour
  contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

  res1 = img.copy()
  count = 0
  for con in contours:
    area = cv2.contourArea(con)
    radian = int(math.sqrt(area / 3.14))
    minRad = int(radian * 0.3)
    maxRad = int(radian * 2)
    mask_temp = np.zeros(mask.shape[:2],dtype=np.uint8)
    cv2.drawContours(mask_temp, [con], 0, (255), -1)  
    circles = cv2.HoughCircles(mask_temp,cv2.HOUGH_GRADIENT,1, 1.2 * radian,
                              param1=100,param2=10,minRadius=minRad,maxRadius=maxRad)
    
    if circles is not None:
      circles = np.uint16(np.around(circles))
      for i in circles[0, :]:
        # circle outline
        radius = i[2]
        if radius > radius_avg:
          count += 1
          center = (i[0], i[1])
        # circle center
          cv2.putText(res1, str(count), center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
          cv2.circle(res1, center, radius, (0, 0, 255), 3)



  print('Number of object is', count)
  APP_ROOT = os.path.dirname(os.path.abspath(__file__))
  OUTPUT_FOLDER = os.path.join(APP_ROOT, 'output/')
  cv2.imwrite(os.path.join(OUTPUT_FOLDER, fileName +'_result.jpg'), res1)
  dictionary = {fileName:count}
  np.save(os.path.join(OUTPUT_FOLDER, fileName +'_result.npy'), dictionary) 
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  