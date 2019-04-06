
SKCNN
====
Copyright (C) 2019 Han-Jing Jiang(jianghanjing17@mails.ucas.ac.cn),Zhu-Hong You(zhuhongyou@xjb.ac.cn),Yu-An Huang


Computational drug repositioning
===

We here propose a drug repositioning computational method combining the techniques of Sigmoid Kernel and Convolutional Neural Network (SKCNN) which is able to learn new features effectively representing drug-disease associations via its hidden layers.

Dataset</br>
--
1.CdrugSimilarity and CdiseaseSimilarity store disease similarity matrix and drug similarity matrix of Cdataset</br>
2.Drugstore similarity and diseaseSimilarity store diseaseSimilarity matrix and drugSimilarity matrix of Fdataset.</br>
3.Drug -disease-whole and c-drug-disease-whole store known drug-disease associations of Cdataset and Fataset.</br>


code</br>
--

1.SigmoidKernel.py:Function to generate SigmoidKernel similarity</br>
2.Feature.py:Function to generate the total characteristics</br>
3.CNN.p;y:The features are obtained by the convolutional neural network</br>
4.RF.py：predict potential indications for drugs</br>
All files of Dataset and Code should be stored in the same folder to run SKCNN.
