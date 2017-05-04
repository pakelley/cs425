#!/usr/bin/python

import os
import time
import datetime
import io
import pypyodbc
import csv
import sys
import struct
import atexit
from signal import SIGTERM
import numpy as np

#---------- GLOBAL SETTINGS VARIABLES ---------------

TIMEINTERVAL = 20
#timeinterval is needed so that the time between
#datetime.now() and the query is not
#infinitesimally small - measured in seconds

BLOBNUMBER = 4 #unused as of yet

#----------------------------------------------------

class SampleRow(object):
    '''Row of Data from Query with Blob Cracking'''
    def __init__(self,segment_id,time,rpm,hexData,crackedData):
        self.segment_id = segment_id
        self.time = time
        self.rpm = rpm
        self.hexData = hexData
        self.crackedData = crackedData
    '''Crack the Hex Data to an array of ints, return into crackedData'''
    def crackHexData(self):
        bytearr = bytearray(self.hexData)
        self.crackedData = []
        i=0
        j=1
        while (j<4096): #4096 bytes makes 2048 16-bit ints
            x1 = hex(bytearr[i])[2:].zfill(2)
            y1 = hex(bytearr[j])[2:].zfill(2)
            v1 = int(str(y1+x1), 16)
            self.crackedData.extend([v1])
            i = i+2
            j = j+2

class RunForever(object):
    def __init__(self):
        self.run()
    def run(self):
        conn = pypyodbc.connect("Driver={SQL Server Native Client 11.0};"
                                "Server=AVANTTH-PC;"
                                "Database=System1_Hist;"
                                "Trusted_Connection=yes;")
        cursor = conn.cursor()

        print "starting"
        count = 0
        while(count < 1):
            classRowList = []
            #List of Each Sample in Query stored in the SampleRow class
            now = datetime.datetime.now() - datetime.timedelta(seconds=220)
            utctime = str(now + datetime.timedelta(hours=7))
            utctime = utctime[:-3]
            qcommand = "select segment_id,gmt,start_rev_rpm,sample_data from dbo.waveform where (segment_id = 218 or segment_id = 244 or segment_id = 270 or segment_id = 296) and gmt > '" + str(utctime) + "' order by gmt"
            cursor.execute(qcommand)
            row = cursor.fetchone()
            rowCount = 0
            while row and rowCount<BLOBNUMBER*4:
                classRow = SampleRow(row[0],row[1],row[2],row[3],0)
                classRow.crackHexData()
                classRow.time = classRow.time - datetime.timedelta(hours=7)
                data = np.array(classRow.crackedData)
                rowTup = [data, classRow.segment_id]
                classRowList.extend([rowTup])
                rowCount = rowCount + 1
                row = cursor.fetchone()
            for p in classRowList: print p
            count = count + 1
        print "done"
        conn.close()

RunForever()
