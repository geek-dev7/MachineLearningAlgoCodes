dataset=read.csv('Salary_Data.csv')
 library(caTools)
set.seed(123)
split=sample.split(dataset$Salary,SplitRatio=2/3)
traning_set=subset(dataset,split=TRUE)
test_set=subset(dataset,split=FALSE)



