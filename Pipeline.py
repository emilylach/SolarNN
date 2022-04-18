import pandas as pd
import datetime as dt
import os
import sys
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
        self.floatdata = pd.DataFrame()
        
        
        self.xtrain = []
        self.ytrain = []
        self.xval = []
        self.yval = []
        self.testx = []
        self.testy = []
        
        self.hours = 60*exp['hours']

     
    def clean_data(self):
        # creating a datetime column
        combinedt = lambda datetime: dt.datetime.strptime(datetime[0]+' '+datetime[1], '%Y/%m/%d %H:%M')
        self.data['datetime'] = self.data[[self.settings['date'],self.settings['time']]].apply(combinedt, axis=1)

        # replace all negatives in data with 0
        for col in self.floatcols:
            self.data[col] = self.data[col].map(neg)

        # add labeled data and drop resulting nan rows
        self.data['label'] = self.data[self.settings['IR']].shift(self.hours)
        self.data.drop(self.data.iloc[0:self.hours].index, inplace=True)
        self.data.reset_index(inplace=True, drop=True)


    def norm(self):
        if self.exp['negone'] == True:
            neg_one = lambda col: (1-(-1))*((col-min(col))/(max(col)-min(col)))+(-1) 
            self.data[self.floatcols] = self.data[self.floatcols].apply(neg_one)
        else:
            zero_one = lambda col: (1-(0))*((col-min(col))/(max(col)-min(col)))+(0) 
            self.data[self.floatcols] = self.data[self.floatcols].apply(zero_one)

            
    def split_label(self):
        """ Dropping nighttime data depending on experiment"""
        if self.exp['dropnight']==True:
            dropidx = list(self.data[self.data['datetime'].dt.hour.isin([0,1,2,3,4,5,20,21,22,23])].index)
            self.data.drop(dropidx, inplace=True)
            self.data.reset_index(inplace=True, drop=True)
        else:
            pass

        """ splitting the data and label """
        self.label = self.data['label']
        # I need to take average of every 10 minutes
        self.data.drop('label',axis=1, inplace=True)
        # I can only put float/int values in NN so am filtering out other ocls
        self.floatdata = self.data[self.floatcols]
    

    def train_val(self, trainsplit=0.7):
        print('i made it to train val')
        # Splitting datasets using a random index value and grabbing 120 values after 

        samples = []
        # length of data - total number of samples to not go out of index
        for i in range(len(self.floatdata)-self.hours):
            # grabbing 120 length samples to take into account 2 hours of data
            samples.append(self.floatdata.values[i:i+self.hours].reshape(-1))
        samples = np.array(samples)
            
        labels = []
        for k in range(len(self.label)-self.hours):
            # grabbing as many 120 samples as possible then flattening them
            labels.append(self.label.values[k:k+self.hours].reshape(-1))
        # I need to take the average over 10 minutes
        labels = np.array(labels)

        # 10 minute increments over the hours of prediction
        labels = labels.reshape(-1, 10, int(self.hours/10)).mean(axis=1)  


        if self.exp['type'] =='test':
            # if test data, it doesn't need to be split
            self.testx = samples
            self.testy = labels

        else:
            # train data needs to be split into train and val
            print('imade it to else statement')
            train_split = round(len(samples)*trainsplit)  # number of samples to allocate to train data
            label_split = round(len(labels)*trainsplit)
            self.xtrain = samples[0:train_split]
            self.xtrain = self.xtrain
            self.xval = samples[train_split:]
            self.xval = self.xval

            self.ytrain = labels[0:label_split]
            self.ytrain = self.ytrain
            self.yval = labels[label_split:]
            self.yval = self.yval
            


    def derivative(self, column):
        colname = 'd{col}'.format(col=column)
        self.data[colname] = self.data[self.settings[column]].diff()/self.data['datetime'].diff().dt.seconds

