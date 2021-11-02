asdf# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 11:51:34 2021
@author: marioantolinezherrer
"""
import pyhrv as hrv
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
print(columns)


#creating dataframes for every data column 
id_col = df['ID'] #patient id
RR_int_col = df['RR Int'] #RR intervals data
type_col = df['Type'] #rearrest type


#Select directory where we are going to store the ouput files & remove previous files
dir = '/Users/marioantolinezherrera/Desktop/CWRU/FALL/EBME_421/course_project/py_codes/hrv_library/hrv_results'
for f in os.listdir(dir):
    os.remove(os.path.join(dir, f))
    
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

counter = 0
previous_item = id_col.get(0)

patient_data.clear()
patients_list.clear()

for i in id_col: #iterates through patient ids column
            
    if(i == previous_item): #if same patient as previous row, append RR_int_col data to patient_data
        if(RR_int_col.get(i) != None):
            patient_data.append(float(RR_int_col.get(i)))
    else: #if different patient as previous row, evaluate function and clear patient_data 
        if(patient_data.__len__() > 1):
            try:
                evaluate_patient_data(patient_data, i)
            except ValueError:
                print('ValueError on index:' + str(counter))
                pass
            patient_data.clear()
        else:
            patients_list.append(i)
            
    previous_item = i
    counter +=1
print('All data has been processed')
sys.exit()
#To do:
    #1) stop if two blank cells are detected
    
    #2) analyse and interpretate results of arrest type with result data
    
#References:
    #https://pyhrv.readthedocs.io/en/latest/_pages/api/hrv.html#ref-hrvfunc 
    #https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html 
    #https://biosppy.readthedocs.io/en/stable/tutorial.html
    #https://www.tutorialspoint.com/How-to-save-a-Python-Dictionary-to-CSV-file
