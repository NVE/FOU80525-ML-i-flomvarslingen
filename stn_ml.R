# Read data from Azure share point 
# setwd('C:/Users/byha/OneDrive - Norges vassdrags- og energidirektorat/Dokumenter/ML')
# edit

library(readxl)
library(openxlsx)
library(zoo)
#---- keras --------
library(tensorflow)
library(keras)
library(tfautograph)
library(reticulate)
library(purrr)
library(data.table)
library(ggplot2)

# setwd("Z:\\pbo\\ML") # pick a working directory
setwd("Z:/GIT-repo")

# Just testing the multi editing 

### read_excel('//companySharepointSite/project/.../ExcelFilename.xlsx', 'Sheet1', skip=1) does not work
# add a comment as pbo
#data<- read_excel('C:/Users/byha/OneDrive - Norges vassdrags- og energidirektorat/Dokumenter/ML/data/Qtd-1-9.xlsx',sheet = 1, skip=0)
# getSheetNames()
# 
# getSheetNames(system.file("exclSheetNames", 'C:/Users/byha/OneDrive - Norges vassdrags- og energidirektorat/Dokumenter/ML/data/Qtd-1-9.xlsx', package = "openxlsx"))
# 
# my.wb <- openXL('C:/Users/byha/OneDrive - Norges vassdrags- og energidirektorat/Dokumenter/ML/data/Qtd-1-9.xlsx')
# 
# names(my.wb)

sheetnam <- '2.323.24h'

# data<- read.xlsx('C:/Users/byha/OneDrive - Norges vassdrags- og energidirektorat/Dokumenter/ML/data/Qtd-1-9.xlsx', sheet = sheetnam)
data<- read.xlsx('Qtd-1-9.xlsx', sheet = sheetnam)


# -----------  for daily data use this 

#  as.Date(paste(data[1,1],data[1,2],data[1,3],sep="-")) 

# [1] "1970-10-10"

names(data)[1:3] <- c("year", "month", "day") # column heading columns 1:3
names(data)[4:13] <- sprintf("rain%d",1:10)   # Column heading for columns 4:13
names(data)[14:23] <- sprintf("temp%d",1:10)   # column headings fr columns 14:23
names(data)[24] <- "flow"                     # column heading for columns 24 
           
data$date <- as.Date(paste(data$year,data$month,data$day,sep="-"), format = "%Y-%m-%d")

dat <- zoo(as.data.frame(cbind(rowMeans(data[,4:13]),rowMeans(data[14:23]), data$flow)),data$date)

names(dat) <-c("P_mm", "T_oC", "Q_m3s")
rm(data)
# plot(dat, main=sheetnam, xlab='')




# ------- Specify and scale the x_train data set and x_test data ---------- 
split_factor=0.8  # use 80 % of all samples in x_train after number of rows
dat=scale(dat)
dat <- dat[,!(colSums(is.na(dat)) > 0)]
indx <- sample(nrow(dat),round(nrow(dat)*split_factor,0)) 

x_train <- as.matrix(dat[indx,-dim(dat)[2]])
x_test <- as.matrix(dat[-indx,-dim(dat)[2]])
y_train <- as.matrix(dat[indx,dim(dat)[2]])
y_test <- as.matrix(dat[-indx,dim(dat)[2]])

#rm(dat,indx)

#------- Normalizing data ------------

# x_train <- scale(x_train)
# x_test <- scale(x_test, center = attr(x_train, "scaled:center"),
#                 scale=attr(x_train, "scaled:scale"))

# plot(x_train[,1], type='l')
# 
# Removal of columns that potentially contain NAs

# x_train <- x_train[,!(colSums(is.na(x_train)) > 0)]
#-------Plot x_test and x_train datasets for visual control-----
tr=as.data.frame(x_train)
tt=as.data.frame(x_test)
ggplot(tr, aes(x=P_mm, y=T_oC))+
  geom_point(data=tr) + geom_point(data=tt, col='red') #+ 
  # theme(legend.position = 'top') + 
  # geom_text(split_factor)
# plot(x_train, col='black')
# plot(x_test, col='red')

# ----------- build mdoel ---------------


build_model <- function() {
  model <- keras_model_sequential() 
  model %>% 
    layer_dense(units = 256, 
                input_shape = dim(x_train)[2],
                kernel_regularizer = regularizer_l2(l = 0.001)) %>% 
    layer_activation_relu() %>% # (2) Separate ReLU layer
    layer_dense(units = 8,
                activation = 'relu') %>% # (1) Specified in the dense layer
    layer_dropout(0.6) %>%
    # layer_dense(units = 128,
    #           activation = 'relu') %>%
    layer_dense(units = 1)
  
  model %>% compile(
    loss = "mse",
    optimizer = optimizer_rmsprop(),
    metrics = list("mean_absolute_error")
  )
  model
}

model <- build_model()
model %>% summary()

# --------------- predict on test data and plot predicted vs observed values ------------------

print_dot_callback <- callback_lambda(
  on_epoch_end = function(epoch, logs) {
    if (epoch %% 80 == 0) cat("\n")
    cat(".")
  }
)

early_stop <- callback_early_stopping(monitor = "val_loss", patience = 20)

epochs <- 100

# Fit the model and store training stats
history <- model %>% fit(
  x_train,
  y_train,
  epochs = epochs,
  validation_split = 0.2,
  verbose = 1,
  callbacks = list(early_stop, print_dot_callback)
)


# - plots 
# -1 x-y plot

y_test_pred <- model %>% predict(x_test)
ggplot(data.table('test' = c(y_test),'pred' = c(y_test_pred)), 
       aes(x=test, y=pred)) + geom_point() + xlim(0,20) + ylim(0,15) +
  geom_smooth(method='lm')


# 2 time series

y_test0 <- dat[-indx,dim(dat)[2]]

res1 <- merge(y_test0, zoo(y_test_pred, index(y_test0)))
names(res1) <-c('obs','pred')

p <- autoplot(res1[365:720,], facet = NULL)

p
# COMMENT: should be scaled back. it is strange with negative values
ggplot(res1, aes(index(res1))) +                    # basic graphical object
  geom_line(aes(y=obs), colour="black") +  # first layer
  geom_line(aes(y=pred), colour="red")  # second layer
