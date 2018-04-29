#preparing the data and making it sequential for RNN


#first converting data to float

#you have to call this function
import csv
import numpy

sequences=[]
labels=[]

with open("data.csv",'r') as f:
    reader=csv.reader(f)
    next(reader)
    for i in reader:
        sequences.append([float(seq) for seq in i[1:]])
        labels.append(i[0])

numpy.save('data',sequences)
numpy.save('labels',labels)






# splitting the data in train and test data and saving in numpy file
#you have to call this function
import numpy as np
import random

cat=np.load('data.npy')
lab=np.load('labels.npy')

def get_train_data():


    train_data=[[[k] for k in i] for i in cat]

    data_final=list(zip(train_data,lab))

    random.shuffle(data_final)
    random.shuffle(data_final)
    train=data_final[:int(len(data_final)*0.8)]
    test= data_final[int(len(data_final)*0.8):]
    print(np.array(train[0][0]).shape)

    np.save('train_data',train)
    np.save('test_data',test)





# loading saved train and test numpy dataset and taking batch size data

import numpy as np
import random
train_data=np.load('train_data.npy')
test_data=np.load('test_data.npy')

def get_train():
    data=train_data

    cat_o=[]
    label=[]

    for i,j in data:
        cat_o.append(i)
        label.append(j)

    return {'input':cat_o,'label':label}

print(get_train()['input'][:2])
print(get_train()['label'][:2])



#get test data


def get_test():
    data=test_data
    cat=[]
    labs=[]

    for i,j in data:
        labs.append(j)
        cat.append(i)

    return {'cat':cat,'labs':labs}
