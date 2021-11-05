# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 11:51:34 2021
@author: marioantolinezherrer
"""
import pyhrv
import biosppy
from csv import reader
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import sys 
import os

filename_xlsx = '/Users/marioantolinezherrera/Desktop/CWRU/FALL/EBME_421/course_project/py_codes/laken_irish_excels/EBME 421 Project - HRV Files.xlsx'

#1. obtaining the data from a single .csv file
# data = list()
# with open(filename, "r") as fd:
#   for line in fd:
#       line = line.strip()
#       if line != 'RR Intervals': #go to second line 
#           data.append(float(line))
          


#2. obtaining the data from the .xlsx file
df = pd.read_excel(filename_xlsx, sheet_name='Sheet1')
columns = df.columns

#creating dataframes for every data column 
id_col = df['ID'] #patient id
RR_int_col = df['RR Int'] #RR intervals data
type_col = df['Type'] #rearrest type


#Select directory where we are going to store the ouput files & remove previous files

#This method creates creates a .csv results file from the patients_data
#counter value is used to assign a new name to each results
def evaluate_patient_data(input_data, ID_col):
    file_creator_index = ID_col;
    hrv_out = hrv.hrv(input_data)  
    dict_out = hrv_out.as_dict()
    csv_columns = dict_out.keys()
    csv_file = dir + '/' + str(file_creator_index) + '.csv'
    file = open(csv_file, 'x')
    with open(csv_file, 'w') as f:
        for key in dict_out.keys():
            f.write("%s,%s\n"%(key,dict_out[key]))
            
            
#reads data from each patient, appends it into a data list (patient_data)
#and evaluates function in a loop over every element in data row
patients_list = list()
patient_data = list()


arrest_type_data = list()
nni_counter_data = list()
hr_mean_data  =list()
sdnn_data = list()
nNN50_data = list()
pNN50_data = list()
rmssd_data = list()

counter = 0
previous_item = 0

patient_data.clear()
patients_list.clear()

#for the first row
patients_list.append(id_col.get(0))
patient_data.append(float(RR_int_col.get(0)))

for i in range(1, len(id_col)): #iterates through patient ids column
            
    if(id_col.get(i) == id_col.get(previous_item)): #if same patient as previous row, append RR_int_col data to patient_data
        if(RR_int_col.get(i) != None):
            patient_data.append(float(RR_int_col.get(i)))
    else: #if different patient as previous row, evaluate function and cli fiear patient_data 
        if(patient_data.__len__() > 1):
            print(i)
            patients_list.append(id_col.get(i))
            try:
                #evaluate_patient_data(patient_data, i)
                #nni_counter
                arrest_type_out = type_col.get(i)
                arrest_type_data.append(arrest_type_out)
                
                nni_parameters_out = pyhrv.time_domain.nni_parameters(patient_data)
                nni_counter_data.append(nni_parameters_out['nni_counter'])
                
                hr_parameters_out = pyhrv.time_domain.hr_parameters(patient_data)
                hr_mean_data.append(hr_parameters_out['hr_mean'])
                
                sdnn_out = pyhrv.time_domain.sdnn(patient_data)
                sdnn_data.append(sdnn_out['sdnn'])
                
                NN50_out = pyhrv.time_domain.nn50(patient_data)
                nNN50_data.append(NN50_out['nn50'])
                pNN50_data.append(NN50_out['pnn50'])
                
                rmssd_out = pyhrv.time_domain.rmssd(patient_data)
                rmssd_data.append(rmssd_out['rmssd'])
                
                
            except ValueError:
                print('ValueError on index:' + str(counter))
                pass
            patient_data.clear()
        else:
            pass
            
    previous_item = i
    counter +=1


df = pd.DataFrame([patients_list, arrest_type_data, nni_counter_data, hr_mean_data, sdnn_data, nNN50_data, pNN50_data, rmssd_data])
df = df.transpose()
df.columns = ['ID','Arrest Type','NNI Counter','HR Mean','SDNN','nNN50','pNN50','RMSSD']

#deletes data in dir, creates output file in dir and stores data in .csv
dir = '/Users/marioantolinezherrera/Desktop/CWRU/FALL/EBME_421/course_project/py_codes/hrv_library/hrv_results'
for f in os.listdir(dir):
    os.remove(os.path.join(dir, f))
    
df.to_csv(dir + '/output_data.csv')
print('All data has been processed')
sys.exit()

    
#References:
    #https://pyhrv.readthedocs.io/en/latest/_pages/api/hrv.html#ref-hrvfunc 
    #https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html 
    #https://biosppy.readthedocs.io/en/stable/tutorial.html
