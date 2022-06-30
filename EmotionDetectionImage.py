import cv2
from deepface import DeepFace
import matplotlib.pyplot as plt
import numpy as np 

imgpath = "C:/Users/HP/Downloads/Alok_Pic.jpg"
image = cv2.imread(imgpath)

# plt.imshow(image[:,:,::-1])
# plt.show()

analyze = DeepFace.analyze(image,actions=['emotion'])  #here the first parameter is the image we want to analyze #the second one there is the action
print(analyze['dominant_emotion'])