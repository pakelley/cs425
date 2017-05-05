import PIL.Image

import os

import time
import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_RIGHT, TA_CENTER, TA_LEFT
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import letter

import io

def producePivot(condVec,rollVec):

	#initialize
	pivotTable = []
	
	#set top row
	pivotTable.append([])
	
	#initialize top row and find safe state column
	safeCol = 0
	for cond in condVec[0]:
		if (cond is "Safe") or (cond is "safe"):
			safeCol = len(pivotTable[0])
			pivotTable[0].append("Unsafe")

		pivotTable[0].append(cond)

	#set first column
	row = 0
	for roll in rollVec[0]:
		
		if row is not 0:
			pivotTable.append([])
			pivotTable[row].append(roll)
		row += 1
	
	#append row for column totals
	pivotTable.append([])
		

	#compute pivot cells
	row = 1
	for roll in rollVec[1]:
		if not isinstance(roll,basestring):
			unsafeTotal = 0
			for cond in condVec[1]:
				if not isinstance(cond,basestring):
					if not (safeCol is (len(pivotTable[row]))):
						unsafeTotal += cond * roll / 100.0					
						pivotTable[row].append( (cond * roll) / 100.00 )
					else:
						pivotTable[row].append(unsafeTotal)
						pivotTable[row].append((cond * roll)/ 100.00)

			row += 1
			
	#compute column 
	pivotTable[row].append("Total")
	for index in range(1,len(pivotTable[0])):
		colTotal = 0;
		for jndex in range(1,len(pivotTable)-1):
			colTotal += pivotTable[jndex][index]
		pivotTable[row].append(colTotal)
			
	return pivotTable
					

def writeReport(fileNames,condVec,rollVec,simplify):
	buf = io.BytesIO()

	doc = SimpleDocTemplate(
		buf,
		rightMargin=inch/2,
		leftMargin = inch/2,
		bottomMargin=inch/2,
		pagesize=letter
	)

	styles = getSampleStyleSheet()
	styles.add(ParagraphStyle(name='modTitle', fontName = 'Helvetica',fontSize=20,backColor = colors.black, textColor=colors.white, alignment=TA_LEFT))

	size = 480,320

	try:
		#resize image
		if not simplify:
			index = 1
			for name in fileNames:
				sImage = PIL.Image.open(name)
				resizedImage = sImage.resize(size)
				resizedImage.save("temp" + str(index) + ".png", "png")
				index += 1

		#create table for both rows and columns
		rcVec = []
		index = 0
		totalRows = max(len(rollVec[0]),len(condVec[0]))
		rSplit = len(rollVec)-1
		cSplit = len(condVec)-1
		
		for row in rollVec:
			rcVec.append([])
			if index == 0:
				rcVec[index].append("Roll States")
			else:
				rcVec[index].append("")

			for data in row:
				rcVec[index].append(data)
			for cell in range(len(rcVec[index]),totalRows):
				rcVec[index].append("")
			index += 1

		rcVec.append([])			
		for row in range(totalRows):
			rcVec[index].append("")

		index += 1
		start = index

		for row in condVec:
			rcVec.append([])
			if index == start:
				rcVec[index].append("Error Modes")
			else:
				rcVec[index].append("")			
			for data in row:
				rcVec[index].append(data)
			for cell in range(len(rcVec[index]),totalRows):
				rcVec[index].append("")
			index += 1
									
		#create reportlab table
		emptyCStart = len(condVec[0])
		emptyRStart = len(rollVec[0])
		rcData = Table(zip(*rcVec),hAlign='CENTER')
		rcData.setStyle(TableStyle([('VALIGN',(0,0),(0,-1),'TOP'),
						('HALIGN',(0,-1),(-1,-1),'LEFT'),
					   ('INNERGRID', (0,1), (rSplit,emptyRStart), 0.25, colors.black),
					   ('INNERGRID', (-1-cSplit,1), (-1,emptyCStart), 0.25, colors.black),
                       ('BOX', (0,1), (rSplit,emptyRStart), 0.25, colors.black),
					   ('BOX', (-1-cSplit,1), (-1,emptyCStart), 0.25, colors.black)
                       ]))
					   
		#produce pivot table
		pVec = producePivot(condVec,rollVec)	
		
		#set up table for pdf
		pData = Table(pVec,hAlign='CENTER')
		pData.setStyle(TableStyle([('VALIGN',(0,0),(0,-1),'TOP'),
						('HALIGN',(0,-1),(-1,-1),'LEFT'),
                       ('INNERGRID', (0,0), (-1,-1), 0.25, colors.black),
                       ('BOX', (0,0), (-1,-1), 0.25, colors.black),
  					   ('INNERGRID',(-2,0),(-2,-1),0.25,colors.red),
					   ('INNERGRID',(-1,0),(-1,-1),0.25,colors.blue),
					   ('BOX',(-2,0),(-2,-1),0.25,colors.red),
					   ('BOX',(-1,0),(-1,-1),0.25,colors.blue),
  					   ('BACKGROUND',(-2,0),(-2,-1),colors.orange),
					   ('BACKGROUND',(-1,0),(-1,-1),colors.turquoise)
                       ]))
		
		#Create sections of pdf document
		sections = []
		
		#add title
		sections.append(	Paragraph("<br/><br/>  Turbine Condition Report - " + 
			datetime.datetime.fromtimestamp(time.time()).strftime('%Y/%m/%d %H:%M:%S') + 
			"<br/><br/><br/>", 
			styles['modTitle'])	)
			
		#add tables
		sections.append(	Paragraph("<br/><br/>	", styles['Normal'])	)
		sections.append(rcData);
		sections.append(	Paragraph("<br/><br/><para alignment=\"center\">Pivot Table	", styles['Normal'])	)
		sections.append(pData)
		sections.append(	Paragraph("<br/><br/>	", styles['Normal'])	)

		#add graph if needed
		if not simplify:
			index = 1
			for name in fileNames:
				sections.append(	Image("temp" + str(index) + ".png")	)
				index += 1
			
		#build document
		doc.build( sections )

	except IOError:
		print("Graph image(s) or temp.png File could not be read/saved\n\tMaybe the file is open somewhere else")

	#save product pdf to file system
	try:
		with open("report.pdf", 'w') as fd:
			fd.write(buf.getvalue())
	except IOError:
		print("report.pdf File could not be saved\n\tMaybe the file is open somewhere else")
		
	#remove resized image from file system
	try:
		if not simplify:
			index = 1
			for name in fileNames:
				os.remove("temp" + str(index) + ".png")
				index += 1
	except OSError:
		print("Unable to remove temporary data file 'temp.png'.\n\tPlease remove file if it still exists in the directory")
		
		
#test program
condVec = [["Classifier Name","rub","preload","outer bearing","inner bearing","Safe"],
		["Confidence Level",12.5,12.5,37.5,25,12.5]]
rollVec = [["Classifier Name","slow roll","ramp up","ramp down"],
		["Confidence Level",12.5,12.5,75]]
fNames = ["motorside_orbit.png","motorside_x.png","motorside_y.png"]
writeReport(fNames,condVec,rollVec,False)