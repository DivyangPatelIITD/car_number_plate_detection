# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 17:09:21 2021

@author: divya
"""
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from keras.models import Sequential,load_model,save_model
from keras.layers import Activation,Dropout,Flatten,Conv2D,MaxPooling2D,Dense,ActivityRegularization
import keras
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Input, Dropout
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential,load_model,save_model
import pandas as pd

def char_contours(dimensions, img) :

    # Find all contours in the image
    contrs= cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

    # setting up the limits of contours
    llimit_w = dimensions[0]
    ulimit_w = dimensions[1]
    llimit_h = dimensions[2]
    ulimit_h = dimensions[3]
    
    
    contrs = sorted(contrs, key=cv2.contourArea, reverse=True)[:15]
    
    ii = cv2.imread('contour.jpg')
    
    x_cntr_list = []
    target_contours = []
    img_res = []
    for cntr in contrs :
        # detects contour in binary image and returns the coordinates of rectangle enclosing it
        intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)
        
        # checking the dimensions of the contour to filter out the characters by contour's size
        if intWidth > llimit_w-3  and intWidth < ulimit_w+15 and intHeight > llimit_h and intHeight < ulimit_h :
            x_cntr_list.append(intX) #stores the x coordinate of the character's contour, to used later for indexing the contours

            char_copy = np.zeros((44,24))
            # extracting each character using the enclosing rectangle's coordinates.
            char = img[intY:intY+intHeight, intX:intX+intWidth]
            char = cv2.resize(char, (20, 40))
            
            cv2.rectangle(ii, (intX,intY), (intWidth+intX, intY+intHeight), (50,21,200), 2)
            #plt.imshow(ii, cmap='gray')
            #plt.title('Character Segments')

            # Make result formatted for classification: invert colors
            char = cv2.subtract(255, char)

            # Resize the image to 24x44 with black border
            char_copy[2:42, 2:22] = char
            char_copy[0:2, :] = 0
            char_copy[:, 0:2] = 0
            char_copy[42:44, :] = 0
            char_copy[:, 22:24] = 0

            img_res.append(char_copy) # List that stores the character's binary image (unsorted)
            
    # Return characters on ascending order with respect to the x-coordinate (most-left character first)
            
    #plt.show()
    # arbitrary function that stores sorted list of character indeces
    indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
    img_res_copy = []
    for idx in indices:
        img_res_copy.append(img_res[idx])# stores character images according to their index
    img_res = np.array(img_res_copy)

    return img_res

def list_of_charac(image) :

    # Preprocess cropped license plate image
    img_lp = cv2.resize(image, (333, 75))
    img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)
    _, img_binary_lp = cv2.threshold(img_gray_lp, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_binary_lp = cv2.erode(img_binary_lp, (3,3))
    img_binary_lp = cv2.dilate(img_binary_lp, (3,3))

    LP_WIDTH = img_binary_lp.shape[0]
    LP_HEIGHT = img_binary_lp.shape[1]

    # make the borders white
    img_binary_lp[0:3,:] = 255
    img_binary_lp[:,0:3] = 255
    img_binary_lp[72:75,:] = 255
    img_binary_lp[:,330:333] = 255

    # Estimations of character contours sizes of cropped license plates
    dimensions = [LP_WIDTH/6,
                       LP_WIDTH/2,
                       LP_HEIGHT/10,
                       2*LP_HEIGHT/3]
    """
    plt.imshow(img_binary_lp, cmap='gray')
    plt.title('Binary image')
    plt.show()
    cv2.imwrite('contour.jpg',img_binary_lp)
    """
    # Get contours within cropped license plate
    char_list = char_contours(dimensions, img_binary_lp)

    return char_list

def fix_dimension(img): 
    new_img = np.zeros((28,28,3))
    for i in range(3):
        new_img[:,:,i] = img
        return new_img
  
def show_results(char,loaded_model):
    dic = {}
    characters = '#0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for i,c in enumerate(characters):
        dic[i] = c

    output = []
    for i,ch in enumerate(char): #iterating over the characters
        img_ = cv2.resize(ch, (28,28), interpolation=cv2.INTER_AREA)
        img = fix_dimension(img_)
        img = img.reshape(1,28,28,3) #preparing image for the model
        #y_ = loaded_model.predict_classes(img)[0] #predicting the class
        x=loaded_model.predict(img)
        #print(x)
        y_=np.argmax(x,axis=1)
        #print()
        #print(y_[0])
        character = dic[y_[0]]
        if(character!='#'):
            output.append(character) #storing the result in a list
        
    plate_number = ''.join(output)
    
    return plate_number



if __name__ == '__main__':
    path = "C:/IIITD_Sem_3/Digital_image_processing/project/Final_images_cropped"
    dir_list = os.listdir(path)
    file_list=[]
    number_list=[]
    for file in dir_list:
        
        file_list.append(file)
        img = cv2.imread(path+'/'+file)#CarLongPlate47 (1).jpg')
        img=cv2.resize(img,(500,300))
        char=list_of_charac(img)
        #print(char[5])
        #img=
        """
        my_img=(char[4]*255).astype(int)
        img_2 = Image.fromarray(my_img)
        img_2.save('img9_1.png')
        
        for i in range(len(char)):
            plt.subplot(1, len(char), i+1)
            plt.imshow(char[i], cmap='gray')
            plt.axis('off')
        plt.show()
        """
        
        loaded_model = Sequential()
        loaded_model.add(Conv2D(16, (22,22), input_shape=(28, 28, 3), activation='relu', padding='same'))
        loaded_model.add(Conv2D(32, (16,16), input_shape=(28, 28, 3), activation='relu', padding='same'))
        loaded_model.add(Conv2D(64, (8,8), input_shape=(28, 28, 3), activation='relu', padding='same'))
        loaded_model.add(Conv2D(64, (4,4), input_shape=(28, 28, 3), activation='relu', padding='same'))
        loaded_model.add(MaxPooling2D(pool_size=(4, 4)))
        loaded_model.add(Dropout(0.4))
        loaded_model.add(Flatten())
        loaded_model.add(Dense(128, activation='relu'))
        loaded_model.add(Dense(37, activation='softmax'))
        
        # Restore the weights
        loaded_model.load_weights('checkpoints_3/my_checkpoint')
        text_num=show_results(char,loaded_model)
        #sing_entry.append(text_num)
        number_list.append(text_num)
        print(text_num)
        #final_list.append(sing_entry)
    dicto = {'File_name': file_list, 'Number ': number_list}  
       
    df = pd.DataFrame(dicto) 
    
# saving the dataframe 
    df.to_csv('detected_nums_5.csv') 
        
    """
    img = cv2.imread('C:/IIITD_Sem_3/Digital_image_processing/project/Final_images_cropped/CarLongPlate522.jpg')#CarLongPlate47 (1).jpg')
    #img = cv2.imread('C:/IIITD_Sem_3/Digital_image_processing/project/Cars39_cropped.png')
    img=cv2.resize(img,(500,300))
    char=list_of_charac(img)
    #print(char[5])
    #img=
    
    my_img=(char[4]*255).astype(int)
    img_2 = Image.fromarray(my_img)
    img_2.save('img9_1.png')
    
    for i in range(len(char)):
        plt.subplot(1, len(char), i+1)
        plt.imshow(char[i], cmap='gray')
        plt.axis('off')
    plt.show()
    
    
    loaded_model = Sequential()
    loaded_model.add(Conv2D(16, (22,22), input_shape=(28, 28, 3), activation='relu', padding='same'))
    loaded_model.add(Conv2D(32, (16,16), input_shape=(28, 28, 3), activation='relu', padding='same'))
    loaded_model.add(Conv2D(64, (8,8), input_shape=(28, 28, 3), activation='relu', padding='same'))
    loaded_model.add(Conv2D(64, (4,4), input_shape=(28, 28, 3), activation='relu', padding='same'))
    loaded_model.add(MaxPooling2D(pool_size=(4, 4)))
    loaded_model.add(Dropout(0.4))
    loaded_model.add(Flatten())
    loaded_model.add(Dense(128, activation='relu'))
    loaded_model.add(Dense(37, activation='softmax'))
    
    # Restore the weights
    loaded_model.load_weights('checkpoints/my_checkpoint')
    print(show_results(char,loaded_model))
    """