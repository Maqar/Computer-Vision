import cv2
import numpy as np

from PIL import Image
from matplotlib import pyplot as plt

def disp(img):
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)

def region_of_interest(image):
    h = image.shape[0]
    w = image.shape[1]
    polygons = np.array([(0, h), (w, h), (w, h//1.5), (0, h//1.5)])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, np.array([polygons], dtype=np.int64), 1024)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def make_coord(image, line_parameters):
    # r = x*cos(t) + y*sin(t)
    k = np.pi/180
    t, r = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * (3/5))
    x1 = int((r - y1*np.sin(t*k))/np.cos(t*k))
    x2 = int((r - y2*np.sin(t*k))/np.cos(t*k))
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):

    left_fit = []
    right_fit = []

    while lines is not None:
        for line in lines:
            t, r = line
            if t < 90:
                left_fit.append((t, r))
            else:
                right_fit.append((t, r))
        #x, y = 160, 153
        if len(left_fit) == 0:
          left_fit.append((47, 221))
        if len(right_fit) == 0:
          right_fit.append((138, -17))

        left_fit_average = np.average(left_fit, axis=0)
        left_line = make_coord(image, left_fit_average)
        right_fit_average = np.average(right_fit, axis=0)
        right_line = make_coord(image, right_fit_average)
        return np.array([left_line, right_line])

def prep_proc(image):
  img = cv2.imread(image)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  edges_0 = cv2.Canny(gray,50,200)
  edges = region_of_interest(edges_0)
  return edges

def hough(image):

  edges = prep_proc(image)

  h = edges.shape[0]
  w = edges.shape[1]
  d = int(np.sqrt(h**2 + w**2))

  k = np.pi/180

  H = np.zeros((180, d*2)) #матрица theta x rho

  for i in range(round(h//1.5), h):
    for j in range(w):
      if edges[i][j] == 255:
        for t in range(0, 180):
          if (t > 3 and t < 70) or (t > 110 and t < 177):
            r = int(round(i*np.sin(t*k) + j*np.cos(t*k)) + d)
            H[t][r] = H[t][r] + 1 

  c_lines = []
  for i in range(H.shape[0]):
    for j in range(H.shape[1]):
      if H[i][j] > 40:
        c_lines.append([i, j-d])

  return c_lines

def lines_point(image):

  c_lines  = hough(image)
  img = cv2.imread(image)
  k = np.pi/180

  for i in range(len(c_lines)): #прямые из c_lines
              t, r = c_lines[i]
              y1 = img.shape[0]
              y2 = int(y1 * (2/5))
              x1 = int((r - y1*np.sin(t*k))/np.cos(t*k))
              x2 = int((r - y2*np.sin(t*k))/np.cos(t*k))
              cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

  lines = average_slope_intercept(img, c_lines) #средня правая и левая линии

  #точка пересечения прямых
  x1,y1,x2,y2 = lines[0]
  k0 = (y2-y1)/(x2-x1)
  b0 = -x1*(y2-y1)/(x2-x1) + y1

  x1,y1,x2,y2 = lines[1]
  k1 = (y2-y1)/(x2-x1)
  b1 = -x1*(y2-y1)/(x2-x1) + y1

  x = float((b1-b0)/(k0-k1))
  y = float(k0*x+b0)

  for i in range(len(lines)):  
    x1,y1,x2,y2 = lines[i]
      #if abs(y2-y1) > 1:
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
  cv2.circle(img, (int(x),int(y)), radius=3, color=(255,255,255), thickness=-1)
  #return disp(img)
  return [x, y]

import os
import json

d = {}

path = "test/"
dirs = os.listdir(path)

for filename in dirs:
    img = os.path.join(path, filename)
    d[filename] = lines_point(img)

with open("markup1.json", "w") as outfile: 
    json.dump(d, outfile)
