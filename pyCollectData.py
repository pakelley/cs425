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

#---------- GLOBAL SETTINGS VARIABLES ----------------------

TIMEINTERVAL = 2
#timeinterval is needed so that the time between
#datetime.now() and the query is not
#infinitesimally small - measured in seconds

BLOBNUMBER = 4
#The number of 4-set sensor blobs allowed
#If BLOBNUMBER = 5, then 20 sensor blobs will be collected

#-----------------------------------------------------------

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


def getUTCtime():
    now = datetime.datetime.now() - datetime.timedelta(seconds=TIMEINTERVAL)
    utctime = str(now + datetime.timedelta(hours=7))
    #utctime = "2017-05-04 21:16:00" #------------ For Testing
    return utctime[:-3]

def executeQuery(cursor, utctime):
    qcommand = "select segment_id,gmt,start_rev_rpm,sample_data from dbo.waveform where (segment_id = 218 or segment_id = 244 or segment_id = 270 or segment_id = 296) and gmt > '" + str(utctime) + "' order by gmt, segment_id"
    cursor.execute(qcommand)

def checkRowLoopValid(rowCount, classRow):
    if  ((rowCount % 4 == 0 and classRow.segment_id != 218) or
        (rowCount % 4 == 1 and classRow.segment_id != 244) or
        (rowCount % 4 == 2 and classRow.segment_id != 270) or
        (rowCount % 4 == 3 and classRow.segment_id != 296)):
        print "INVADLID DATA-------------------------------------------- CONSECUTIVE IDs ---------------"
        return False
    return True

class RunForever(object):
    def __init__(self):
        self.run()
    def run(self):
        conn = pypyodbc.connect("Driver={SQL Server Native Client 11.0};"
                                "Server=AVANTTH-PC;"
                                "Database=System1_Hist;"
                                "Trusted_Connection=yes;")
        cursor = conn.cursor()
        if not cursor:
            print "Cannot Connect to SQL Database - Exiting Program"
            return
        else:
            print "program starting"

        toggle = True
        while(1):

            '''
            -----------WHAT EVERYONE ELSE NEEDS-------------
            ONLY np arrays of each sensor blob
            ALWAYS will be a multiple of 4
            ALWAYS will be ordered in ascending segment_id's
            1st = sid 218 - Motor Side X-Sensor
            2nd = sid 244 - Motor Side Y-Sensor
            3rd = sid 270 - Outer Side X-Sensor
            4th = sid 296 - Outer Side Y-Sensor
            '''
            dataPassingList = []

            #FULL information on each sensor blob row
            classRowList = [] #WHAT BRIAN NEEDS

            #Gets UTC time from Actual Time for use in SQL Query
            utctime = getUTCtime()

            #Runs the query
            executeQuery(cursor, utctime)

            row = cursor.fetchone()
            if row:
                #print " "
                #print "Query FULL"
                #print " "
                toggle = True

            rowCount = 0
            valid = True

            while row and rowCount<BLOBNUMBER*4 and valid:
                classRow = SampleRow(row[0],row[1],row[2],row[3],0)
                classRow.crackHexData()
                classRow.time = classRow.time - datetime.timedelta(hours=7)
                data = np.array(classRow.crackedData)

                valid = checkRowLoopValid(rowCount, classRow)

                sensorBlob = [data, classRow.segment_id]
                classRowList.extend([sensorBlob])
                rowCount = rowCount + 1
                #writer.writerow([rowTup])
                row = cursor.fetchone()

            if not classRowList and toggle:
                #print " "
                #print "QUERY EMPTY"
                #print " "
                toggle = False
            elif rowCount % 4 != 0:
                print "INVADLID DATA-------------------------------------------- NOT GROUPED BY 4 ---------------"
            else:
                for p in classRowList:
                    dataPassingList.extend([p[0]])
                    #print p[0]


            '''


            CLASSIFIER CODE


            '''


        print "program ending"
        conn.close()


#Run Daemon
RunForever()
