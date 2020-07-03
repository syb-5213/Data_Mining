import numpy as np 
import pandas as pd 
import math 
import matplotlib.pyplot as plt 
from keras.layers import LSTM
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from sklearn.metrics import mean_squared_error
from keras.layers import Dropout
import keras
from scipy.optimize import curve_fit

#加载数据
data = pd.read_csv('us-counties.csv')

#数据格式转换
t_data='2020-01-21'
num=1
sc=-1
sd=0
sum_data=[]
for index, row in data.iterrows():
    sc=sc+row["cases"]
    sd=sd+row["deaths"]
    if t_data!=row["date"]:
        t_data=row["date"]
        sum_data.append((num,sc,sd))
        num=num+1
        sc=0
        sd=0
sin_data=[]
sin_data.append(sum_data[0])
for i in range(len(sum_data)-1):
    sin_data.append((sum_data[i+1][0],sum_data[i+1][1]-sum_data[i][1],sum_data[i+1][2]-sum_data[i][2]))
sum_data = pd.DataFrame(sum_data,columns=['Day','Confirmed','Deaths'])    
sum_data.to_csv("sum_data.csv",index=False)
sin_data = pd.DataFrame(sin_data,columns=['Day','Confirmed','Deaths'])    
sin_data.to_csv("sin_data.csv",index=False)       
        

#回归模型
def logistic_model(x,a,b,c):
    return c/(1+np.exp(-(x-b)/a))
def exponential_model(x, a, b, c):
    return a * np.exp(b * (x - c))
#处理确诊数据集
x =sum_data["Day"].tolist()
y =sum_data["Confirmed"].tolist()
fit1 = curve_fit(logistic_model,x,y)
y_pred_logistic=[logistic_model(i,fit1[0][0],fit1[0][1],fit1[0][2]) for i in x]
fit2 = curve_fit(exponential_model,x,y,maxfev=500000)
plt.plot(x,y, label="True data" )
plt.plot(x,[exponential_model(i,fit2[0][0],fit2[0][1],fit2[0][2]) for i in x], label="Exponential result" )
plt.plot(x,[logistic_model(i,fit1[0][0],fit1[0][1],fit1[0][2]) for i in x], label="Logistic result" )
plt.legend()
plt.title('Confirmed')
plt.savefig('Confirmed.jpg')
plt.show()
y_pred_exponential=[exponential_model(i,fit2[0][0],fit2[0][1],fit2[0][2]) for i in x]
print("logistic_model rmse is:"+str(math.sqrt(mean_squared_error(y_pred_logistic,y))))
print("exponential_model rmse is:"+str(math.sqrt(mean_squared_error(y_pred_exponential,y))))
#处理死亡数据集
x =sum_data["Day"].tolist()
y =sum_data["Deaths"].tolist()
fit1 = curve_fit(logistic_model,x,y)
y_pred_logistic=[logistic_model(i,fit1[0][0],fit1[0][1],fit1[0][2]) for i in x]
fit2 = curve_fit(exponential_model,x,y,maxfev=500000)
plt.plot(x,y, label="True data" )
plt.plot(x,[exponential_model(i,fit2[0][0],fit2[0][1],fit2[0][2]) for i in x], label="Exponential result" )
plt.plot(x,[logistic_model(i,fit1[0][0],fit1[0][1],fit1[0][2]) for i in x], label="Logistic result" )
plt.legend()
plt.title('Deaths')
plt.savefig('Deaths.jpg')
plt.show()
y_pred_exponential=[exponential_model(i,fit2[0][0],fit2[0][1],fit2[0][2]) for i in x]
print("logistic_model rmse is:"+str(math.sqrt(mean_squared_error(y_pred_logistic,y))))
print("exponential_model rmse is:"+str(math.sqrt(mean_squared_error(y_pred_exponential,y))))



#lstm模型
     
#数据预处理
def create_dataset(dataset,step,validate_rate=0.67):
    dataX,dataY=[],[]
    trainX,trainY,testX,testY=[],[],[],[]
    for i in range(len(dataset)-step):
        a = dataset[i:(i+step), 0]
        dataX.append(a)
        dataY.append(dataset[i+step, 0])
    trainX=dataX[:int(validate_rate*len(dataX))]
    trainY=dataY[:int(validate_rate*len(dataY))]
    testX=dataX[int(validate_rate*len(dataX)):]
    testY=dataY[int(validate_rate*len(dataY)):]
    return np.array(trainX), np.array(trainY),np.array(testX),np.array(testY)

#记录loss曲线
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.xlabel(loss_type)
        plt.ylabel('loss')
        plt.legend(loc="upper right")
        plt.show()  

#处理确诊数据集
dataset=sin_data["Confirmed"].to_frame()
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
train = dataset
step = 40
validate_rate=0.8
trainX, trainY, testX, testY = create_dataset(train,step,validate_rate)
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
history = LossHistory()
model = Sequential()
model.add(LSTM(100, input_shape=(1, step)))
model.add(Dropout(0.15))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')#使用均方差做损失函数。优化器用adam
model.fit(trainX, trainY, epochs=200, validation_split=0.1,batch_size=32, verbose=1,callbacks=[history])#训练模型，100epoch，批次为1，每一个epoch显示一次日志，学习率动态减小
history.loss_plot("batch")

testY = scaler.inverse_transform([testY])
trainPredict=model.predict(trainX)
a=np.append(scaler.inverse_transform(trainX[0]),scaler.inverse_transform(trainPredict))

temp=testX[0].reshape(1,testX.shape[1],testX.shape[2])
result=[]
for i in range(len(testY[0])):
    pre=model.predict(temp)[0][0]
    result.append(pre)
    temp=temp[0][0][1:]
    temp=np.append(temp,pre)
    temp=temp.reshape(1,1,temp.shape[0])
result=np.array(result)
result=result.reshape(np.array(result).shape[0],1)
predict=scaler.inverse_transform(result)    

plt.plot(scaler.inverse_transform(dataset),label='true')
plt.plot(a,label='trainpredict')
plt.plot(range(144-len(predict),144),predict,label='testpredict')
plt.title('Confirmed')
plt.legend()
plt.show()

plt.plot(range(0,21),scaler.inverse_transform(dataset)[144-len(predict):144],label='true')
plt.plot(range(0,21),predict,label='testpredict')
plt.legend()
plt.title('Confirmed')
plt.show()
TestScore = math.sqrt(mean_squared_error(testY[0,:], predict))
print('Test Score: %.2f RMSE' % (TestScore))


#处理死亡数据集
dataset=sin_data["Deaths"].to_frame()
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
train = dataset
step =40
validate_rate=0.8
trainX, trainY, testX, testY = create_dataset(train,step,validate_rate)
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
history = LossHistory()
model = Sequential()
model.add(LSTM(25, input_shape=(1, step)))
model.add(Dropout(0.15))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')#使用均方差做损失函数。优化器用adam
model.fit(trainX, trainY, epochs=100, validation_split=0.1,batch_size=32, verbose=1,callbacks=[history])#训练模型，100epoch，批次为1，每一个epoch显示一次日志，学习率动态减小
history.loss_plot("batch")

testY = scaler.inverse_transform([testY])
trainPredict=model.predict(trainX)
a=np.append(scaler.inverse_transform(trainX[0]),scaler.inverse_transform(trainPredict))

temp=testX[0].reshape(1,testX.shape[1],testX.shape[2])
result=[]
for i in range(len(testY[0])):
    pre=model.predict(temp)[0][0]
    result.append(pre)
    temp=temp[0][0][1:]
    temp=np.append(temp,pre)
    temp=temp.reshape(1,1,temp.shape[0])
result=np.array(result)
result=result.reshape(np.array(result).shape[0],1)
predict=scaler.inverse_transform(result)

plt.plot(scaler.inverse_transform(dataset),label='true')
plt.plot(a,label='trainpredict')
plt.plot(range(144-len(predict),144),predict,label='testpredict')
plt.title('Deaths')
plt.legend()
plt.show()
   
plt.plot(range(0,21),scaler.inverse_transform(dataset)[144-len(predict):144],label='true')
plt.plot(range(0,21),predict,label='testpredict')
plt.legend()
plt.title('Deaths')
plt.show()
TestScore = math.sqrt(mean_squared_error(testY[0,:], predict))
print('Test Score: %.2f RMSE' % (TestScore))
