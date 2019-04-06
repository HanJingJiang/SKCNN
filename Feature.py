import numpy as np
np.random.seed(1337)
import csv
from numpy import *
import SigmoidKernel
def ReadMyCsv(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:
        SaveList.append(row)
    return

def storFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return


OriginalData = []
ReadMyCsv(OriginalData, "c-drug-disease-whole.csv")
print(len(OriginalData))

counter = 0
while counter < len(OriginalData):
    OriginalData[counter][0] = OriginalData[counter][0].lower()
    OriginalData[counter][1] = OriginalData[counter][1].lower()
    counter = counter + 1
LncDisease = []
counter = 0
while counter < len(OriginalData):
    Pair = []
    Pair.append(OriginalData[counter][0])
    Pair.append(OriginalData[counter][1])
    LncDisease.append(Pair)
    counter = counter + 1
storFile(LncDisease, 'cLncDisease.csv')

AllDisease = []
counter1 = 0
while counter1 < len(OriginalData): 
    counter2 = 0
    flag = 0
    while counter2 < len(AllDisease):  
        if OriginalData[counter1][1] != AllDisease[counter2]:
            counter2 = counter2 + 1
        elif OriginalData[counter1][1] == AllDisease[counter2]:
            flag = 1
            counter2 = counter2 + 1
    if flag == 0:
        AllDisease.append(OriginalData[counter1][1])
    counter1 = counter1 + 1
print('len(AllDisease)', len(AllDisease))
storFile(AllDisease, 'cAllDisease.csv')
# 构建AllDRUG
AllDRUG = []
counter1 = 0
while counter1 < len(OriginalData): 
    counter2 = 0
    flag = 0
    while counter2 < len(AllDRUG): 
        if OriginalData[counter1][0] != AllDRUG[counter2]:
            counter2 = counter2 + 1
        elif OriginalData[counter1][0] == AllDRUG[counter2]:
            flag = 1
            break
    if flag == 0:
        AllDRUG.append(OriginalData[counter1][0])
    counter1 = counter1 + 1
DiseaseAndDrugBinary = []
counter = 0
while counter < len(AllDisease):
    row = []
    counter1 = 0
    while counter1 < len(AllDRUG):
        row.append(0)
        counter1 = counter1 + 1
    DiseaseAndDrugBinary.append(row)
    counter = counter + 1


print('len(LncDisease)', len(LncDisease))
counter = 0
while counter < len(LncDisease):
    DN = LncDisease[counter][1]
    RN = LncDisease[counter][0]
    counter1 = 0
    while counter1 < len(AllDisease):
        if AllDisease[counter1] == DN:
            counter2 = 0
            while counter2 < len(AllDRUG):
                if AllDRUG[counter2] == RN:
                    DiseaseAndDrugBinary[counter1][counter2] = 1
                    break
                counter2 = counter2 + 1
            break
        counter1 = counter1 + 1
    counter = counter + 1
lines = [line.strip().split() for line in open("CdiseaseSimilarity.txt")]
txtSimilarity = []
i = 0
for dis in lines:
    i = i + 1
    if i == 1:
        continue
    txtSimilarity.append(dis[1:])

lines = [line.strip().split() for line in open("CdrugSimilarity.txt")]
drugtxtSimilarity = []
i = 0
for dis in lines:
    i = i + 1
    if i == 1:
        continue
    drugtxtSimilarity.append(dis[1:])
print('len(drugtxtSimilarity)',len(drugtxtSimilarity))
DiseasePolynomialKernel1 = SigmoidKernel.SigmoidKernelDisease(DiseaseAndDrugBinary)
storFile(DiseasePolynomialKernel1, "cDiseasePolynomialKernel.csv")
print("disease")
DrugPolynomialKernel1 = SigmoidKernel.SigmoidKernelRNA(DiseaseAndDrugBinary)
storFile(DrugPolynomialKernel1, "cDrugPolynomialKernel.csv")
import random
counter1 = 0  
counter2 = 0    
counterP = 0    
counterN = 0    
PositiveSample = []     
PositiveSample = LncDisease
print('PositiveSample)', len(PositiveSample))

NegativeSample = []
counterN = 0
while counterN < len(PositiveSample):                     
    counterD = random.randint(0, len(AllDisease)-1)
    counterR = random.randint(0, len(AllDRUG)-1)   
    DiseaseAndRnaPair = []
    DiseaseAndRnaPair.append(AllDRUG[counterR])
    DiseaseAndRnaPair.append(AllDisease[counterD])
    flag1 = 0
    counter = 0
    while counter < len(LncDisease):
        if DiseaseAndRnaPair == LncDisease[counter]:
            flag1 = 1
            break
        counter = counter + 1
    if flag1 == 1:
        continue
    flag2 = 0
    counter1 = 0
    while counter1 < len(NegativeSample):
        if DiseaseAndRnaPair == NegativeSample[counter1]:
            flag2 = 1
            break
        counter1 = counter1 + 1
    if flag2 == 1:
        continue
    if (flag1 == 0 & flag2 == 0):
        NegativePair = []
        NegativePair.append(AllDRUG[counterR])
        NegativePair.append(AllDisease[counterD])
        NegativeSample.append(NegativePair)
        counterN = counterN + 1
print('len(NegativeSample)', len(NegativeSample))
DiseaseSimilarity = []
counter = 0
while counter < len(AllDisease):
    counter1 = 0
    Row = []
    while counter1 < len(AllDisease):
        v = float(DiseasePolynomialKernel1[counter][counter1])
        if v > 0:
            Row.append(v)
        if v == 0:
            Row.append(txtSimilarity[counter][counter1])
        counter1 = counter1 + 1
    DiseaseSimilarity.append(Row)
    counter = counter + 1
print('len(DiseaseSimilarity)', len(DiseaseSimilarity))
print('len(DiseaseSimilarity[0])',len(DiseaseSimilarity[0]))
storFile(DiseaseSimilarity, 'cDiseaseSimilarity.csv')

DRUGSimilarity = []
counter = 0
while counter < len(AllDRUG):
    counter1 = 0
    Row = []
    while counter1 < len(AllDRUG):
        v = float(DrugPolynomialKernel1[counter][counter1])
        if v > 0:
            Row.append(v)
        if v == 0:
            Row.append(drugtxtSimilarity[counter][counter1])
        counter1 = counter1 + 1
    DRUGSimilarity.append(Row)
    counter = counter + 1
print('len(DRUGSimilarity)', len(DRUGSimilarity))
print('len(DRUGSimilarity[0)',len(DRUGSimilarity[0]))
storFile(DRUGSimilarity, 'cDRUGSimilarity.csv')

AllSample = PositiveSample.copy()
AllSample.extend(NegativeSample)        

# SampleFeature
SampleFeature = []
counter = 0
while counter < len(AllSample):
    counter1 = 0
    while counter1 < len(AllDRUG):
        if AllSample[counter][0] == AllDRUG[counter1]:
            a = []
            counter3 = 0
            while counter3 <len(DRUGSimilarity[0]):
                v = DRUGSimilarity[counter1][counter3]
                a.append(v)
                counter3 = counter3 + 1
            break
        counter1 = counter1 + 1
    counter2 = 0
    while counter2 < len(AllDisease):
        if AllSample[counter][1] == AllDisease[counter2]:
            b = []
            counter3 = 0
            while counter3 < len(DiseaseSimilarity[0]):
                v = DiseaseSimilarity[counter2][counter3]
                b.append(v)
                counter3 = counter3 + 1
            break
        counter2 = counter2 + 1
    a.extend(b)
    SampleFeature.append(a)
    counter = counter + 1
counter1 = 0
storFile(SampleFeature, 'cSampleFeature.csv')
