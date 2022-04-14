import random
import pandas as pd
import datetime as dt
import os
import sys
import random
import numpy as np


def neg(a):
    """ replaces negative values with zero """
    if a < 0:
        return 0
    else:
        return a


class DataClean:
    """ Class to clean solar radiation data """
    def __init__(self, settings, exp):
        self.exp = exp
        try:
            if self.exp['type'] == 'train':
                self.data = pd.read_csv('./archive/train.csv')
            elif self.exp['type'] == 'test':
                self.data = pd.read_csv('./archive/test.csv')
            else: 
                print('There must be a "type":"train/test" in experiements')
        except:
            print('cwd in path:{inpath}'.format(inpath=os.getcwd() in sys.path))
            print('If false, you are not in the right parent folder, must be in SolarNN')
        
        self.data = self.data[self.exp['columns']]
        self.settings = settings
        self.floatcols = self.data.select_dtypes(include=[float,int]).columns
        self.label = pd.Series()
        
        
        self.xtrain = []
        self.ytrain = []
        self.xval = []
        self.yval = []
        self.testx = []
        self.testy = []
            
     
    def clean_data(self):
        # creating a datetime column
        combinedt = lambda datetime: dt.datetime.strptime(datetime[0]+' '+datetime[1], '%Y/%m/%d %H:%M')
        self.data['datetime'] = self.data[[self.settings['date'],self.settings['time']]].apply(combinedt, axis=1)

        # replace all negatives in data with 0
        for col in self.floatcols:
            self.data[col] = self.data[col].map(neg)

        # add labeled data and drop resulting nan rows
        self.data['label'] = self.data[self.settings['IR']].shift(120)
        self.data.drop(self.data.iloc[0:120].index, inplace=True)
        self.data.reset_index(inplace=True, drop=True)


    def norm(self):
        if self.exp['negone'] == True:
            neg_one = lambda col: (1-(-1))*((col-min(col))/(max(col)-min(col)))+(-1) 
            self.data[self.floatcols] = self.data[self.floatcols].apply(neg_one)
        else:
            zero_one = lambda col: (1-(0))*((col-min(col))/(max(col)-min(col)))+(0) 
            self.data[self.floatcols] = self.data[self.floatcols].apply(zero_one)

            
    def split_label(self):
        """ splitting the data and label """
        self.label = self.data['label']
        self.data.drop('label',axis=1, inplace=True)
        # I can only put float/int values in NN so am filtering out other ocls
        self.data = self.data[self.floatcols]


    def itterdata(self, num_of_samples, numsamples):
        for i in range(num_of_samples):
            yield self.data.sample(numsamples)
        

    def itterlabel(self, num_of_samples, numsamples):
        for i in range(num_of_samples):
            yield self.label.sample(numsamples)


    def train_val(self, hours_per_sample=2, num_of_samples=526920, trainsplit=0.7):
        print('i made it to train val')
        # Splitting datasets using a random index value and grabbing 120 values after 
        numsamples = 60 * hours_per_sample
        
        samples = pd.concat(self.itterdata(num_of_samples, numsamples))
        labels =  pd.concat(self.itterlabel(num_of_samples, numsamples))


        if self.exp['type'] =='test':
            # if test data, it doesn't need to be split
            self.testx = samples
            self.testy = labels

        else:
            # train data needs to be split into train and val
            print('imade it to else statement')
            train = int(num_of_samples*trainsplit)  # number of samples to allocate to train data
            self.xtrain = samples[0:train]
            self.xval = samples[train:]

            self.ytrain = labels[0:train]
            self.yval = labels[train:0]
            


    def derivative(self, column):
        colname = 'd{col}'.format(col=column)
        self.data[colname] = self.data[self.settings[column]].diff()/self.data['datetime'].diff().dt.seconds

