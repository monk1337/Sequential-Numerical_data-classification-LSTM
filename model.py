import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import data_preparing


#categories
classes= ['A','f']

#batch_size
batch_size=50

#iteration
iteration=50//50

#epoch
epoch=200



class RNN_classifier():

    def __init__(self):


        #placeholders

        input_x = tf.placeholder(dtype=tf.float32,shape=[None, 457, 1])    # [None,457,1]  represent  [batch_size, No_of_sequence , n_input ]
        labels = tf.placeholder(dtype=tf.int32,shape=[None,] )             # [labels ]
        self.play={'input':input_x,'out':labels}




        # input_fr = tf.unstack(input_x, 457, 1)     #5X1


        #LSTM model

        with tf.variable_scope('encoder') as scope:

            # we are using bidirectional dynamic rnn here which have forward cell and backword cell

            #forward cell
            fr_cell = rnn.LSTMCell(num_units=250)

            #backward cell
            bw_cell = rnn.LSTMCell(num_units=250)


            #dropout for not making it bised model

            #forward cell dropout
            fr_dropout = rnn.DropoutWrapper(cell=fr_cell, output_keep_prob=0.5)

            #backward cell dropout
            bw_dropout = rnn.DropoutWrapper(cell=bw_cell, output_keep_prob=0.5)


            #LSTM model
            model = tf.nn.bidirectional_dynamic_rnn(fr_dropout, bw_dropout, inputs=input_x, dtype=tf.float32)


        #from RNN we will get two output one is final output and other is first and last state output


        #output is final output and fs and fc are first and last state output , we need final output so we will use output only


        output, (fs, fc) = model

        #output  5x457x250


        #tranforming the data because we will receive time major output from RNN but we want batch major output
        fr_rnn = tf.transpose(output[0], [1, 0, 2])   #457x5x250

        bw_rnn = tf.transpose(output[1], [1, 0, 2])   #457x5x250





        final_output = tf.concat([fr_rnn[-1], bw_rnn[-1]], axis=-1)      #5x500

        #fs  5x250

        state_output= tf.concat([fs.c,fc.c],axis=-1)  #5x500

        #weights for fully connected layer
        weights=tf.get_variable('var1',shape=[2*250,2],dtype=tf.float32,initializer=tf.random_uniform_initializer(-0.01,0.01))


        #bias for fully connected layer

        bias= tf.get_variable('var2',shape=[2],dtype=tf.float32,initializer=tf.random_uniform_initializer(-0.01,0.01))


        #linear activation layer

        logits= tf.matmul(final_output,weights) + bias




        #evaluate
        #making it normalize
        prob=tf.nn.softmax(logits)

        #picking up the highest value index_no
        pred=tf.argmax(prob,axis=-1)

        # cross_entropy

        #loss function
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)

        #reducing the loss
        loss = tf.reduce_mean(ce)


        #accuracy
        accuracy=tf.reduce_mean(
                                 tf.cast(
                                          tf.equal(

                                              tf.cast(pred,tf.int32),labels),tf.float32))








        #train  using Adam optimizer , Other options are Gradient Descent etc
        train=tf.train.AdamOptimizer().minimize(loss)



        #training parameters
        self.out={'logits':logits,'prob':prob,'pred':pred,'ce':ce,'loss':loss,'accuracy':accuracy,'train':train,'model':model}



        #using model
        self.evalute={'pred':pred,'prob':prob,'logits':logits}













        # self.out = {'model':model,'each_cell':fs.c,'output':output,'trans':fr_rnn,'state':state_output,'fina':final_output}



def execute_model(model):
    with tf.Session() as sess:


        #intializing all variables

        sess.run(tf.global_variables_initializer())



        #training the model

        for i in range(epoch):
            for j in range(iteration):


                #collecting the data

                cat=data_preparing.get_train()['input']

                #collecting the labels
                lab=data_preparing.get_train()['label']


                lab=[classes.index(i) for i in lab]


                out=sess.run(model.out,feed_dict={model.play['input']:cat,model.play['out']:lab})

                print(out['loss'],out['accuracy'],j,"epoch",i)


        #evalute the model  / Testing the model with test data in same session

        data=data_preparing.get_test()
        input_data=data['cat']
        labels=data['labs']
        for i,j in zip(input_data,labels):
            out=sess.run(model.evalute,feed_dict={model.play['input']:[i]})

            print(out['prob'],classes[out['pred'][0]],'vs',j)







if '__main__'== __name__:

    model=RNN_classifier()

    print(execute_model(model))






#training
#output
0.074349955 0.98 0 epoch 155
# 0.07963486 0.96 0 epoch 156
# 0.07626302 0.98 0 epoch 157
# 0.06660479 0.98 0 epoch 158
# 0.078285486 0.98 0 epoch 159
# 0.06658502 0.98 0 epoch 160
# 0.06895793 0.98 0 epoch 161
# 0.05209667 1.0 0 epoch 162
# 0.06157341 1.0 0 epoch 163
# 0.06955947 0.98 0 epoch 164
# 0.056209568 1.0 0 epoch 165
# 0.06760145 0.98 0 epoch 166
# 0.056373615 0.98 0 epoch 167
# 0.04702667 1.0 0 epoch 168
# 0.06148951 0.98 0 epoch 169
# 0.047084823 0.98 0 epoch 170
# 0.053266387 1.0 0 epoch 171
# 0.04663007 0.98 0 epoch 172
# 0.040377542 1.0 0 epoch 173
# 0.04839473 1.0 0 epoch 174
# 0.04616747 1.0 0 epoch 175
# 0.047333967 0.98 0 epoch 176
# 0.054682184 0.98 0 epoch 177
# 0.05360978 0.96 0 epoch 178
# 0.043833207 1.0 0 epoch 179
# 0.061951112 0.98 0 epoch 180
# 0.036001384 1.0 0 epoch 181
# 0.033399783 1.0 0 epoch 182
# 0.03619152 1.0 0 epoch 183
# 0.031570856 1.0 0 epoch 184
# 0.054344602 0.96 0 epoch 185
# 0.052337326 0.96 0 epoch 186
# 0.039056756 0.98 0 epoch 187
# 0.051473435 0.96 0 epoch 188
# 0.046014618 0.98 0 epoch 189
# 0.03339572 1.0 0 epoch 190
# 0.044339262 0.98 0 epoch 191
# 0.031593915 1.0 0 epoch 192
# 0.0556867 0.98 0 epoch 193
# 0.045619845 0.98 0 epoch 194
# 0.055950306 0.98 0 epoch 195
# 0.042946182 0.98 0 epoch 196
# 0.038708996 0.98 0 epoch 197


#accuracy   98%  



#Testing model on testing data



# [[0.01667206 0.98332787]] f vs f                 #correct prediction 
# [[9.9943298e-01 5.6698837e-04]] A vs A            #correct prediction 
# [[0.99369895 0.006301  ]] A vs f                  #wrong prediction 
# [[0.03356597 0.96643406]] f vs f                   #correct prediction 
# [[9.9982953e-01 1.7040595e-04]] A vs A             #correct prediction 
# [[0.92333275 0.07666729]] A vs A                    #correct prediction 
# [[0.99537885 0.0046212 ]] A vs A                    #correct prediction 
# [[0.87899536 0.12100458]] A vs A                     #correct prediction 
# [[0.0096073 0.9903927]] f vs f                       #correct prediction 
# [[0.998933   0.00106699]] A vs A                     #correct prediction 
# [[9.9962020e-01 3.7983368e-04]] A vs A               #correct prediction 
# [[0.00233427 0.99766576]] f vs f                      #correct prediction 
# [[0.99573046 0.00426953]] A vs A                       #correct prediction 







