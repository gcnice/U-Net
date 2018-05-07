# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 11:41:40 2018

@author: cao
"""

import numpy as np 
import os
import glob
import time,threading

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

'''
该类为数据处理，用于创建训练集，测试集数据，以及加载训练集，测试集。
'''
class dataProcess(object):

    '''
    初始化数据
    '''
    def __init__(self, out_rows, out_cols, data_path = "data\\train\\image", label_path = "data\\train\\label", test_path = "data\\test", npy_path = "npydata", img_type = "png"):

        self.out_rows = out_rows
        self.out_cols = out_cols
        self.data_path = data_path
        self.label_path = label_path
        self.img_type = img_type
        self.test_path = test_path
        self.npy_path = npy_path
        
    '''
    创建训练集
    '''
    def create_train_data(self):
        
        imgs = glob.glob(self.data_path+"\*."+self.img_type)
        labels = glob.glob(self.label_path+"\*."+self.img_type)
       
        '''
        创建训练集的图片与标签数组
        '''        
        imgdatas = np.ndarray((len(imgs),self.out_rows,self.out_cols,1), dtype=np.uint8)
        imglabels = np.ndarray((len(imgs),self.out_rows,self.out_cols,1), dtype=np.uint8)
        
        '''
        将训练集的图片变成向量
        '''   
        image_index = 0        
        for imgname in imgs:
        
            '''
            加载图片
            '''           
            midname = imgname[imgname.rindex("\\")+1:]
            img = load_img(self.data_path +"\\" + midname,grayscale = True)
            
            '''
            图片变成向量，存入数组
            '''              
            img = img_to_array(img)
            imgdatas[image_index] = img

            image_index += 1
            
        '''
        将训练集的标签变成向量
        '''   
        image_index = 0
        for imgname in labels:
            
            '''
            加载图片
            '''           
            midname = imgname[imgname.rindex("\\")+1:]
            label = load_img(self.label_path +"\\" + midname,grayscale = True)
            
            '''
            图片变成向量，存入数组
            '''              
            label = img_to_array(label)
            imglabels[image_index] = label
    
            image_index += 1 
            
        '''
        将所有向量保存
        ''' 
        np.save(self.npy_path + '\\imgs_train.npy', imgdatas)
        np.save(self.npy_path + '\\imgs_mask_train.npy', imglabels)

    '''
    创建测试集
    '''        
    def create_test_data(self):
        
        '''
        创建测试集的数据与标签
        '''         
        imgs = glob.glob(self.test_path+"\*."+self.img_type)
        imgdatas = np.ndarray((len(imgs),self.out_rows,self.out_cols,1), dtype=np.uint8)
        
        '''
        将所有图片变成向量
        '''          
        index = 0
        for imgname in imgs:
            midname = imgname[imgname.rindex("\\")+1:]
            img = load_img(self.test_path + "\\" + midname,grayscale = True)
            img = img_to_array(img)
            imgdatas[index] = img
            index += 1
 
        '''
        将所有向量保存
        ''' 
        np.save(self.npy_path + '\\imgs_test.npy', imgdatas)

    '''
    加载训练集，转化为float32格式，并归一化，对标签数据进行二值化
    '''  
    def load_train_data(self):
    
        imgs_train = np.load(self.npy_path+"\\imgs_train.npy")
        imgs_mask_train = np.load(self.npy_path+"\\imgs_mask_train.npy")
        imgs_train = imgs_train.astype('float32')
        imgs_mask_train = imgs_mask_train.astype('float32')
        imgs_train /= 255
        imgs_mask_train /= 255
        imgs_mask_train[imgs_mask_train > 0.5] = 1
        imgs_mask_train[imgs_mask_train <= 0.5] = 0
        return imgs_train,imgs_mask_train

    '''
    加载测试集，转化为float32格式，并归一化
    '''          
    def load_test_data(self):
        imgs_test = np.load(self.npy_path+"\\imgs_test.npy")
        imgs_test = imgs_test.astype('float32')
        imgs_test /= 255
        return imgs_test
  
    
'''
主函数，创建窗口并进行侦听动作
'''  
def main():
    mydata = dataProcess(512,512)
    time1 = time.clock()
    t1 = threading.Thread(target=mydata.create_train_data)
    t2 = threading.Thread(target=mydata.create_test_data)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    time2 = time.clock()
    print(time2-time1)

'''
函数入口
'''    
if __name__ == '__main__':
    main()