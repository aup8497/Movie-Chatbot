# Movie-Chatbot
A chatbot that is built on seq2seq model using tensorflow on movie dialog corpus :)

## Requirements:
	-Python3.x
	-tensorflow
	-nltk
	-numpy
	-pickle
	-re

####Note: After running data_helper.py data.pkl,vocab.pkl and training_samples.pkl will be created.

	1. data.pkl - This is a dictionary which contains questions and answers sentences in index of numbers of the form => {'questions':[[12,1412,35,23,3,124,35],[32,42,6,73,13,564,2,67],....],'answers'=[[312,1,643,26,41,32,6,100],[643,168,6,3,12],......]}

	2. vocab.pkl - This contains dictionary which contains word2id and id2word dictionaries which map word from index and index to word

	3. training_samples.pkl - this contains the training samples in the form of list of list containing questions and corresponding answer sentences in their id form, i.e  [ [q1,a1] , [q2,a2] , ,....] 
		where,
		q1 = [1,66,234]
		a1 = [34,321,1231]

		q2 = [56,35,5,314]
		a2 = [34,123,12,4532,67]

