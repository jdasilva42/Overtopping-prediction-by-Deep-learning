# -*- coding: utf-8 -*-
'''
ANN prediction overtopping

Module contains ANN prediction functions

    _initilization():
        Initializes the inputs from excel sheet contained to the directory.
        Compute the inputs required for the ANN prediction and normalized them.
        
      
    _prediction():
        Load the ANN model 'ANN_model.h5' and predicted the overtopping.The 
        inverse tranformation is executed.
        
    _control_plot():
        plot the prediction on the valisation area of the ANN model. Used for 
        control the breakwater design parameters.
            
'''

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from tensorflow.keras.models import load_model


class ANN_overtopping():
    
    def __init__(self):
               
        self.Path = os.getcwd()
        
        self.inputs = pd.read_excel(self.Path+'\\'+'Inputs.xlsx')
                   
        self.normalization = joblib.load(self.Path+'\\'+'Normalization.pkl')
        
        self.model = load_model('ANN_model.h5')
        
        self._initialization()
        
        self._prediction()
        
        self.inputs['q_prediction'] = self.q_overtopping
        
        self.inputs.to_excel(self.Path+'\\'+'Overtopping.xlsx')
        
        self._control_plot()
        
        
        
    def _initialization(self):
        
        Lm0 = ((9.8*(self.inputs['Tp toe'])**2)/(2*np.pi))
        data_init = pd.DataFrame({
            'Hm0/Lm0':self.inputs['Hm0 toe']/Lm0,
            'b':self.inputs['b'],
            'h/Lm0':self.inputs['h']/Lm0,
            'ht/Hm0':self.inputs['ht']/self.inputs['Hm0 toe'],
            'Bt/Lm0':self.inputs['Bt']/Lm0,
            'hb/Hm0':self.inputs['hb']/self.inputs['Hm0 toe'],
            'B/Lm0':self.inputs['B']/Lm0,
            'Ac/Hm0':self.inputs['Ac']/self.inputs['Hm0 toe'],
            'Rc/Hm0':self.inputs['Rc']/self.inputs['Hm0 toe'],
            'Gc/Lm0':self.inputs['Gc']/Lm0,
            'm':self.inputs['m'],
            'cotad':self.inputs['cotad'],
            'cotaincl':self.inputs['cotaincl'],
            'gf':self.inputs['gf']
            })
        
        data_init_scaler = self.normalization['input_scaler'].transform(data_init)
        self.data_init_scaler = data_init_scaler
        
        
    def _prediction(self):
        
        prediction_scaler = self.model.predict(self.data_init_scaler)
        q_overtopping = self._transform(prediction_scaler)
        self.q_overtopping= q_overtopping
                
        
       
    def _transform(self, prediction_scaler):
        
        Hm0 = self.inputs['Hm0 toe']
        
        prediction = self.normalization['q_scaler'].inverse_transform(prediction_scaler.reshape(-1,1))
        q_overtopping = (10**(prediction.flatten()))*np.sqrt(9.8*(Hm0)**3)
        return q_overtopping
    
    
    def _control_plot(self):
        
        database = pd.read_excel(self.Path+'\\'+'Database_CLASH.xls')
        
        delete_list = ['Name',
               'Hm0 deep',
               'Tp deep',
               'Tm deep',
               'Tm-1,0 deep',
               'h deep',
               'Tm toe',
               'Tm-1,0 toe',
               'cotau',
               'cotaexcl',
               'tanaB',
               'Pow', 
               'Remark',
               'Reference']
        
        for x in delete_list:
            database = database.drop([x],axis=1)
        
        database.drop(database.index[database['RF']== 4], inplace = True)
        database.drop(database.index[database['CF']== 4], inplace = True)
        database = database.astype('float')
        database = database.dropna()
        database.drop(database.index[database['q'] < 0.000001], inplace=True)

        
        fig = plt.figure(figsize=(8,8))
        plt.scatter(database['Rc']/database['Hm0 toe'],
                    database['q']/((9.8*(database['Hm0 toe'])**3))**(1/2),
                    s =10,marker = '.',c='gray', label = 'Train set')
        plt.scatter(self.inputs['Rc']/self.inputs['Hm0 toe'],
                    self.inputs['q_prediction']/((9.8*(self.inputs['Hm0 toe'])**3))**(1/2),
                    s = 10
                    ,marker = '*',c='k', label = 'Predictions')
        
        plt.yscale('log')
        plt.axis([-1, 10, 0.00000001, 1])
        plt.grid('True'), plt.legend()
        plt.xlabel('Rc/Hm0'), plt.ylabel('q/(9.8*Hm0^3)^(1/2)')
        
        plt.savefig(self.Path+'\\'+'Output_figure.png')    

