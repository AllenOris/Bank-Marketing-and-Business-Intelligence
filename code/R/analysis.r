
bank_additional_full_df<-read.csv('./data/bank-additional/bank-additional-full.csv',sep=';')
bank_additional_df=read.csv('./data/bank-additional/bank-additional.csv',sep=';')
bank_full_df=read.csv('./data/bank/bank-full.csv',sep=';')
bank_df=read.csv('./data/bank/bank.csv',sep=';')
dat=bank_additional_full_df

colnames(dat)
str(dat)

indx<-apply(dat,2,function(x) any(is.na(x)))
indx
dat[1:5,1:12]
dat[1:5,13:21]
dim(dat)

library(pastecs)
library(ggplot2)
library(dfexplore)
# stat.desc(bank_additional_full_df)
# dfplot(dat)

library(ggplot2)
dt=data.frame(A=c(sum(dat$y=='yes'),sum(dat$y=='no')),B=c('yes','no'))
myLabel = as.vector(dt$B)   ## 转成向量，否则图例的标签可能与实际顺序不一致
myLabel
myLabel = paste(myLabel, "(", round(dt$A / sum(dt$A) * 100, 2), "%)        ", sep = "")   ## 用 round() 对结果保留两位小数
myLabel
p <- ggplot(dt, aes(x = "", y = A, fill = B)) + 
  geom_bar(stat = "identity", width = 1) +    
  coord_polar(theta = "y") + 
  labs(x = "", y = "", title = "") + 
  theme(axis.ticks = element_blank()) + 
  theme(legend.title = element_blank(), legend.position = "top") + 
  scale_fill_discrete(breaks = dt$B, labels = myLabel)   ## 将原来的图例标签换成现在的myLabel
p+ggtitle('接受推销比例')


p<-ggplot(dat)+geom_histogram(aes(x=emp.var.rate,fill=y))
p+xlab('就业变动率')+ylab('数量')+ggtitle('就业变动率直方图')

# previous属性直方图(联系次数)
# pdf('img/img01.pdf',width=4,height=4)
# theme(text = element_text(family='GB1'))
p<-ggplot(dat)+geom_histogram(aes(x=previous,fill=y))
p+xlab('联系一个客户的次数')+ylab('数量')+ggtitle('就业变动率直方图')
# +  theme(text = element_text(family='GB1'))
# dev.off()

# previous属性直方图(联系次数)
# pdf('img/img01.pdf',width=4,height=4)
# theme(text = element_text(family='GB1'))
p<-ggplot(dat)+geom_histogram(aes(x=age,width=1))
p+xlab('年龄')+ylab('数量')+ggtitle('年龄分布')

library(pastecs)
stat.desc(dat$duration/60)
ggplot(dat, aes(x=duration)) + geom_density(colour='orange')

myfunc <- function(x){
  sum(x=='yes')/length(x)
}
dat2<-dat
dat2$duration[dat2$duration>1500]<-1500
dat2$duration<-dat2$duration%/%30
dur<-aggregate(dat2$y, by=list(dat2$duration),FUN=myfunc)
library(ggplot2)
# library(lubridate) 
p<-ggplot(data = dur, mapping = aes(x = Group.1, y = x, group = 1000)) + geom_line(
    colour='red')
p+xlab('电话持续时长(30s)')+ylab("接受产品比例")+ggtitle('电话时间与接受比例折线图')

colnames(dat)

dat_e<-dat[,-11]
colnames(dat_e)

library(ggplot2)
myfunc <- function(x){
  sum(x=='yes')/length(x)
}

dat2<-dat
job<-aggregate(dat2$y, by=list(dat2$job),FUN=myfunc)
library(ggplot2)
# library(lubridate) 

ggplot(dat, aes(x=job,fill=y)) + geom_bar(stat="count")+coord_flip()

p<-ggplot(data = job, mapping = aes(x = Group.1, y = x)) + geom_line(
    colour='red')+geom_bar(stat='identity',fill='orange')+coord_flip()
p+xlab('工作')+ylab("接受产品比例")+ggtitle('工作与接受比例比例图')


library(ggplot2)
myfunc <- function(x){
  sum(x=='yes')/length(x)
}

dat2<-dat
marital<-aggregate(dat2$y, by=list(dat2$marital),FUN=myfunc)
library(ggplot2)
# library(lubridate) 

ggplot(dat, aes(x=marital,fill=y)) + geom_bar(stat="count")+xlab('婚姻状况')+ylab("接受产品比例")+ggtitle('婚姻状况与接受比例图')


library(ggplot2)
myfunc <- function(x){
  sum(x=='yes')/length(x)
}

dat2<-dat
education<-aggregate(dat2$y, by=list(dat2$education),FUN=myfunc)
library(ggplot2)
# library(lubridate) 

ggplot(dat, aes(x=education,fill=y)) + geom_bar(stat="count")

p<-ggplot(data = education, mapping = aes(x = Group.1, y = x)) + geom_line(
    colour='red')+geom_bar(stat='identity',fill='orange')
p+xlab('教育状况')+ylab("接受产品比例")+ggtitle('教育状况与接受比例图')+coord_flip()


colnames(dat)

dat_num<-dat[,c('y','age','duration','campaign','pdays','previous',
                'emp.var.rate','cons.price.idx','cons.conf.idx',
                'euribor3m','nr.employed')]
cor(dat_num[,-1])

# data_info<-data.frame(sapply(dat,as.factor))
# # data_info<-as.data.frame(dat)
# # head(data_info)
# head(data_info)
# data_info$y<-as.vector(data_info$y)
# data_info$y[data_info$y=='yes']<-'1'
# data_info$y[data_info$y=='no']<-'0'
# i<-1
# n=length(data_info)
# for(col in data_info[-n]){
#     i<-i+1
#     print(colnames(data_info)[i])
#     print(IV(X=col,Y=data_info$y))
#     cat('\n')
# }

library(InformationValue)
dat_cat<-dat[,c('y','job','marital','education','default','housing',
               'loan','contact','month','day_of_week','poutcome')]
options(scipen = 999, digits = 4)

dat_cat$y<-as.vector(dat_cat$y)
dat_cat$y[dat_cat$y=='yes']<-'1'
dat_cat$y[dat_cat$y=='no']<-'0'
i<-1
IV_info<-data.frame(features=0,InformationValue=0)
IV_info<-IV_info[-1,0]
for(col in dat_cat[-1]){
    i<-i+1
    iv<-IV(X=col,Y=dat_cat$y)
    iv<-data.frame(iv)[1,1]
    IV_info<-rbind(IV_info,data.frame(features=colnames(dat_cat)[i],InformationValue=iv))
    cat('\n')
}
write.csv(IV_info,'result/完整数据集-离散变量IV值')
IV_info

train<-data.frame(dat_cat,dat_num[,-1])
train$y<-as.factor(train$y)
head(train)
str(train)

library(lattice)
library(caret)
train2<-train
train2$y<-as.vector(train2$y)
train2$y[train2$y=="0"]<-'no'
train2$y[train2$y=="1"]<-'yes'
train2$y<-factor(train2$y,levels=c('yes','no'))
levels(train2$y)
train<-train2

set.seed(233)
head(train)
train<-subset(train,select=(c(y,job,education,default,contact,month,poutcome,housing,
                            age,duration,campaign,pdays,previous,emp.var.rate,cons.price.idx,
                            cons.conf.idx,euribor3m,nr.employed)))
head(train)

multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  library(grid)
 
  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)
 
  numPlots = length(plots)
 
  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                    ncol = cols, nrow = ceiling(numPlots/cols))
  }
 
 if (numPlots==1) {
    print(plots[[1]])
 
  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))
 
    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))
 
      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}


# 训练集分割
partition_indexed<-createDataPartition(train$y,times=1,p=.7,list=FALSE)
bank.train<-train[partition_indexed,]
bank.test<-train[-partition_indexed,]

library(ggplot2)
dt<-data.frame(A=c(sum(bank.train$y=='yes'),sum(bank.train$y=='no')),B=c('yes','no'))
myLabel = as.vector(dt$B)   ## 转成向量，否则图例的标签可能与实际顺序不一致
myLabel = paste(myLabel, "(", round(dt$A / sum(dt$A) * 100, 2), "%)        ", sep = "")   ## 用 round() 对结果保留两位小数
p <- ggplot(dt, aes(x = "", y = A, fill = B)) + 
  geom_bar(stat = "identity", width = 1) +    
  coord_polar(theta = "y") + 
  labs(x = "", y = "", title = "") + 
  theme(axis.ticks = element_blank()) + 
  theme(legend.title = element_blank(), legend.position = "top") + 
  scale_fill_discrete(breaks = dt$B, labels = myLabel)   ## 将原来的图例标签换成现在的myLabel
p1<-p+ggtitle('训练集接受推销比例')

dt<-data.frame(A=c(sum(bank.test$y=='yes'),sum(bank.test$y=='no')),B=c('yes','no'))
myLabel = as.vector(dt$B)   ## 转成向量，否则图例的标签可能与实际顺序不一致
myLabel = paste(myLabel, "(", round(dt$A / sum(dt$A) * 100, 2), "%)        ", sep = "")   ## 用 round() 对结果保留两位小数
p <- ggplot(dt, aes(x = "", y = A, fill = B)) + 
  geom_bar(stat = "identity", width = 1) +    
  coord_polar(theta = "y") + 
  labs(x = "", y = "", title = "") + 
  theme(axis.ticks = element_blank()) + 
  theme(legend.title = element_blank(), legend.position = "top") + 
  scale_fill_discrete(breaks = dt$B, labels = myLabel)   ## 将原来的图例标签换成现在的myLabel
p2<-p+ggtitle('测试集接受推销比例')
multiplot(p1, p2,cols=2)

matrix1 <- data.frame(model=0,Accuracy=0,Sensitivity=0,Precision=0,Specificity=0,Recall=0,F1=0,Kappa=0,TP=0,FN=0,FP=0,TF=0)
matrix1 <- matrix1[-1,]
library(miscTools)
saveMatrix1<-function(mat,label){
    df<-data.frame(mat[2])
    df3<-data.frame(mat[3])
    acc<-df3[1,1]
    df4<-data.frame(mat[4])
    sen<-df4[1,1]
    spe<-df4[2,1]
    pre<-df4[5,1]
    rec<-df4[6,1]
    f1<-df4[7,1]
    kappa<-df3[2,1]
    line<-data.frame(model=label,Accuracy=acc,Sensitivity=sen,Precesion=pre,Specificity=spe,Recall=rec,F1=f1,
                     Kappa=kappa,TP=df[1,3],FN=df[2,3],FP=df[3,3],TF=df[4,3])
    return(line)
}

matrix2 <- data.frame(model=0,Accuracy=0,Sensitivity=0,Precision=0,Specificity=0,Recall=0,F1=0,Kappa=0,TP=0,FN=0,FP=0,TF=0)
matrix2 <- matrix2[-1,]
library(miscTools)
saveMatrix2<-function(mat,label){
    df<-data.frame(mat[2])
    df3<-data.frame(mat[3])
    acc<-df3[1,1]
    df4<-data.frame(mat[4])
    sen<-df4[1,1]
    spe<-df4[2,1]
    pre<-df4[5,1]
    rec<-df4[6,1]
    f1<-df4[7,1]
    kappa<-df3[2,1]
    line<-data.frame(model=label,Accuracy=acc,Sensitivity=sen,Precesion=pre,Specificity=spe,Recall=rec,F1=f1,
                     Kappa=kappa,TP=df[1,3],FN=df[2,3],FP=df[3,3],TF=df[4,3])
    return(line)
}

bank.train2<-bank.train
bank.test2<-bank.test
bank.train2$duration<-NULL
bank.test2$duration<-NULL

# train with logistic regression
fit_log_with<-train(y~.,data=bank.train,
              method='LogitBoost',family='binomial')
predictions<-predict(fit_log_with,bank.test[,-1])
mat<-confusionMatrix(predictions,bank.test[,1])
mat
matrix1<-rbind(matrix1,saveMatrix1(mat,"逻辑回归"))

# cut duration
fit_log_without<-train(y~.,data=bank.train2,
              method='LogitBoost',family='binomial')
predictions<-predict(fit_log_without,bank.test2[,-1])
mat<-confusionMatrix(predictions,bank.test2[,1])
mat
matrix2<-rbind(matrix2,saveMatrix2(mat,"逻辑回归"))

# library(e1071)
# tuned<-tune(svm,y~.,
#            data=bank.train,
#            kernel='polynomial',
#            ranges=list(cost=c(0.001,0.01,0.1,1,10,100)))
# summary(tuned)

# svm
library('e1071')
svmfit<-svm(y~.,data=bank.train,kernel='polynomial',cost=100,scale=FALSE)
# summary(svmfit)
predictions.svm<-predict(svmfit,bank.test[,-1],type='class')
mat<-confusionMatrix(predictions.svm,bank.test[,1])
mat
matrix1<-rbind(matrix1,saveMatrix1(mat,"SVM-多项式核函数"))

# tuned<-tune(svm,y~.,
#            data=bank.train,
#            kernel='linear',
#            ranges=list(cost=c(0.001,0.01,0.1,1,10,100)))
# summary(tuned)

# svm
library('e1071')
svmfit<-svm(y~.,data=bank.train,kernel='linear',cost=.1,scale=FALSE)
# summary(svmfit)
predictions.svm<-predict(svmfit,bank.test[,-1],type='class')
mat<-confusionMatrix(predictions.svm,bank.test[,1])
mat
matrix1<-rbind(matrix1,saveMatrix1(mat,"SVM-线性核函数"))

# tuned<-tune(svm,y~.,
#            data=bank.train2,
#            kernel='linear',
#            ranges=list(cost=c(0.01,0.1,1,10)))
# summary(tuned)

# svm
library('e1071')
svmfit<-svm(y~.,data=bank.train2,kernel='linear',cost=1,scale=FALSE)
# summary(svmfit)
predictions.svm<-predict(svmfit,bank.test2[,-1],type='class')
mat<-confusionMatrix(predictions.svm,bank.test2[,1])
mat
matrix2<-rbind(matrix2,saveMatrix2(mat,"SVM-线性核函数"))

# tuned<-tune(svm,y~.,
#            data=bank.train2,
#            kernel='polynomial',
#            ranges=list(cost=c(0.01,0.1,1,10)))
# summary(tuned)

# svm
library('e1071')
svmfit<-svm(y~.,data=bank.train2,kernel='polynomial',cost=.001,scale=FALSE)
# summary(svmfit)
predictions.svm<-predict(svmfit,bank.test2[,-1],type='class')
mat<-confusionMatrix(predictions.svm,bank.test2[,1])
mat
matrix2<-rbind(matrix2,saveMatrix2(mat,"SVM-多项式核函数"))

head(bank.train2)

dummy<-dummyVars(~.,data=bank.train2[,-1])
dummy_train<-predict(dummy,bank.train2[,-1])
dummy_train<-data.frame(dummy_train)
dummy_train$y<-factor(bank.train2$y,levels=c('yes','no'))

dummy<-dummyVars(~.,data=bank.test2[,-1])
dummy_test<-predict(dummy,bank.test2[,-1])
dummy_test<-data.frame(dummy_test)
dummy_test$y<-factor(bank.test2$y,levels=c('yes','no'))

library('e1071')
svmfit<-svm(y~.,data=dummy_train,kernel='polynomial',cost=.001,scale=FALSE)
# summary(svmfit)
n<-length(dummy_test)
predictions.svm<-predict(svmfit,dummy_test[,1:(n-1)],type='class')
confusionMatrix(predictions.svm,dummy_test[,n])

# nnet
nnfit<-train(y~.,data=bank.train,method='nnet',trace=FALSE)
# summary(nnfit)
predictions<-predict(nnfit,bank.test[,-1])
mat<-confusionMatrix(predictions,bank.test[,1])
mat
matrix1<-rbind(matrix1,saveMatrix1(mat,"神经网络-nnet"))

# nnet
nnfit<-train(y~.,data=bank.train2,method='nnet',trace=FALSE)
# summary(nnfit)
predictions<-predict(nnfit,bank.test2[,-1])
mat<-confusionMatrix(predictions,bank.test2[,1])
mat
matrix2<-rbind(matrix2,saveMatrix2(mat,"神经网络-nnet"))

# nnet
nnfit<-train(y~.,data=bank.train,method='multinom',trace=FALSE)
# summary(nnfit)
predictions<-predict(nnfit,bank.test[,-1])
mat<-confusionMatrix(predictions,bank.test[,1])
mat
matrix1<-rbind(matrix1,saveMatrix1(mat,'神经网络-multinom'))

# multinom
nnfit<-train(y~.,data=bank.train2,method='multinom',trace=FALSE)
# summary(nnfit)
predictions<-predict(nnfit,bank.test2[,-1])
mat<-confusionMatrix(predictions,bank.test2[,1])
mat
matrix2<-rbind(matrix2,saveMatrix2(mat,"神经网络-multinom"))

library(mvtnorm)
library(party)
fit_tree<-ctree(y~.,data=bank.train)
plot(fit_tree)

library(rpart)
tree_fit<-rpart(y~.,data=bank.train,method = 'class')
fit.pruned<-prune(tree_fit,cp=tree_fit$cptable[which.min(tree_fit$cptable[,'xerror']),'CP'])
predictions<-predict(fit.pruned,bank.test,type='class')
mat<-confusionMatrix(predictions,bank.test[,1])
mat
matrix1<-rbind(matrix1,saveMatrix1(mat,"分类决策树"))

library(rpart)
tree_fit2<-rpart(y~.,data=bank.train2,method = 'class')
fit.pruned<-prune(tree_fit2,cp=tree_fit2$cptable[which.min(tree_fit2$cptable[,'xerror']),'CP'])
predictions<-predict(fit.pruned,bank.test2,type='class')
mat<-confusionMatrix(predictions,bank.test2[,1])
mat
matrix2<-rbind(matrix2,saveMatrix2(mat,"分类决策树"))

fit.ctree<-ctree(y~.,data=bank.train)
predictions<-predict(fit.ctree,bank.test)
mat<-confusionMatrix(predictions,bank.test[,1])
mat
matrix1<-rbind(matrix1,saveMatrix1(mat,"分类条件推理树"))

fit.ctree<-ctree(y~.,data=bank.train2)
predictions<-predict(fit.ctree,bank.test2)
mat<-confusionMatrix(predictions,bank.test2[,1])
mat
matrix2<-rbind(matrix2,saveMatrix2(mat,"分类条件推理树"))

library(randomForest, warn.conflicts = FALSE)
fit_rf_with<-randomForest(y~.,data=bank.train)
predictions<-predict(fit_rf_with,bank.test)
mat<-confusionMatrix(predictions,bank.test[,1])
mat
matrix1<-rbind(matrix1,saveMatrix1(mat,"随机森林"))

library(randomForest, warn.conflicts = FALSE)
fit_rf_without<-randomForest(y~.,data=bank.train2)
predictions<-predict(fit_rf_without,bank.test2)
mat<-confusionMatrix(predictions,bank.test2[,1])
mat
matrix2<-rbind(matrix2,saveMatrix2(mat,"随机森林"))

matrix1
matrix2

names(matrix1) <- c("模型名称", "准确度", "灵敏度","精度","特异度","召回率","F1","Kappa","True Positive","False Negative",
              "False Positive","True Negative")

names(matrix2) <- c("模型名称", "准确度", "灵敏度","精度","特异度","召回率","F1","Kappa","True Positive","False Negative",
              "False Positive","True Negative")

write.csv(matrix1,'result/完整数据集训练结果-有duration特征.csv')
write.csv(matrix2,'result/完整数据集训练结果-无duration特征.csv')

matrix1

library(DALEX)
performance.test<-bank.test
performance.test$y<-as.vector(performance.test$y)
performance.test$y[performance.test$y=='yes']<-0
performance.test$y[performance.test$y=='no']<-1
performance.test$y=as.numeric(performance.test$y)
performance.test2<-bank.test2
performance.test2$y<-as.vector(performance.test2$y)
performance.test2$y[performance.test2$y=='yes']<-0
performance.test2$y[performance.test2$y=='no']<-1
performance.test2$y=as.numeric(performance.test2$y)

explainer_log_with <- DALEX::explain(fit_log_with, label="LogitBoost", 
                                    data = performance.test, 
                                     y = performance.test$y
                                    ,type='class')
mp_log_with<-model_performance(explainer_log_with)
vi_log_with<-variable_importance(explainer_log_with,
                                 loss_function=loss_root_mean_square)
plot(vi_log_with)

explainer_log_without <- DALEX::explain(fit_log_without, 
                                        label="LogitBoost", 
                                    data = performance.test2, 
                                     y = performance.test2$y
                                    ,type='class')
mp_log_without<-model_performance(explainer_log_without)
vi_log_without<-variable_importance(explainer_log_without,
                                 loss_function=loss_root_mean_square)
plot(vi_log_without)

explainer_rf_with <- DALEX::explain(fit_rf_with, label="ranger", 
                                    data = performance.test, 
                                     y = performance.test$y
                                    ,type='class')
mp_rf_with<-model_performance(explainer_rf_with)
vi_rf_with<-variable_importance(explainer_rf_with,
                                 loss_function=loss_root_mean_square)
plot(vi_rf_with)


library(pdp)
pdp_rf_with  <- variable_response(explainer_rf_with, 
                                  variable =  "age", 
                                  type = "pdp")
plot(pdp_rf_with)

explainer_rf_without <- DALEX::explain(fit_rf_without, label="ranger", 
                                    data = performance.test2, 
                                     y = performance.test2$y
                                    ,type='class')
mp_rf_without<-model_performance(explainer_rf_without)
vi_rf_without<-variable_importance(explainer_rf_without,
                                 loss_function=loss_root_mean_square)
plot(vi_rf_without)

library(pdp)
pdp_rf_without  <- variable_response(explainer_rf_without, variable =  "euribor3m", type = "pdp")
plot(pdp_rf_without)

# summary(fit_rf_with)
fit_rf_with$importance
# summary(fit_rf_without)
fit_rf_without$importance

write.csv(fit_rf_with$importance
,'result/完整数据集-随机森林结果-特征重要程度-有duration.csv')
write.csv(fit_rf_without$importance
,
'result/完整数据集-随机森林结果-特征重要程度-无duration.csv')

importance<-fit_rf_with
library(ggplot2)
summary(dat)

# head(dat[1:10],20)
# head(dat[11:21],20)
mock<-data.frame(age=20,job='student',marital='single',education='university.degree',
                default='no',housing='no',loan='no',
                contact='cellular',month='jul',day_of_week='fri',
                duration=600,campaign=1,pdays=999,previous=0,
                poutcome='success',
                emp.var.rate=1.8,
                cons.price.idx=90,
                cons.conf.idx=-30,
                euribor3m=1.649,
                nr.employed=4000)

mock[,1:12]
mock[,13:20]
mock<-subset(mock,select=c(job,education,default,contact,month,
                               poutcome,housing,age,campaign,pdays,
                               previous,emp.var.rate,cons.price.idx,
                               cons.conf.idx,euribor3m,nr.employed))
mock <- rbind(bank.train2[1, -1] , mock)[-1,]
    
predictions<-predict(fit_rf_without,mock)
as.vector(predictions)

explainer_log_without <- DALEX::explain(fit_rf_without, 
                                        label="randomForest", 
                                    data = mock, 
                                     y = 1
                                    ,type='class')

# explainer_log_without
mp_log_without<-model_performance(explainer_log_without)
1-data.frame(mp_log_without)[,1]
vi_log_without<-variable_importance(explainer_log_without,
                                 loss_function=loss_root_mean_square)

# head(dat[1:10],20)
# head(dat[11:21],20)
mock<-data.frame(age=80,job='retired',marital='married',education='basic.4y',
                default='no',housing='no',loan='no',
                contact='cellular',month='mar',day_of_week='fri',
                duration=600,campaign=5,pdays=6,previous=5,
                poutcome='success',
                emp.var.rate=-1.8,
                cons.price.idx=90,
                cons.conf.idx=-30,
                euribor3m=1.649,
                nr.employed=4000)

mock[,1:12]
mock[,13:20]

mock<-subset(mock,select=c(job,education,default,contact,month,
                               poutcome,housing,age,campaign,pdays,
                               previous,emp.var.rate,cons.price.idx,
                               cons.conf.idx,euribor3m,nr.employed))
mock <- rbind(bank.train2[1, -1] , mock)[-1,]
    
# head(mock)
predictions<-predict(fit_rf_without,mock)
as.vector(predictions)

explainer_log_without <- DALEX::explain(fit_rf_without, 
                                        label="randomForest", 
                                    data = mock, 
                                     y = 1
                                    ,type='class')

# explainer_log_without
mp_log_without<-model_performance(explainer_log_without)
1-data.frame(mp_log_without)[,1]
vi_log_without<-variable_importance(explainer_log_without,
                                 loss_function=loss_root_mean_square)


