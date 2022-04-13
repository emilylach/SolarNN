import pandas as pd
import datetime as dt


def neg(a):
    """ replaces negative values with zero """
    if a < 0:
        return 0
    else:
        return a


class DataClean:
    """ Class to clean solar radiation data """
    def __init__(self, data, settings, exp):
        self.data = data
        self.exp = exp
        self.data = self.data[self.exp['columns']]
        self.settings = settings
        self.floatcols = self.data.select_dtypes(include=[float,int]).columns
        self.label = pd.Series()
            

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


    def derivative(self, column):
        colname = 'd{col}'.format(col=column)
        self.data[colname] = self.data[self.settings[column]].diff()/\
            self.data['datetime'].diff().dt.seconds

