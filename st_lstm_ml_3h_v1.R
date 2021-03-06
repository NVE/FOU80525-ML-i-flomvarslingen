st_lstm_run <- function(data.dir=file.path("Z:","GIT-repo"),fileN='2.463.3h.ptq', tstep='3h',  zones='single'){

  # tstep = '3h' or '24h'
  # zone = 'single' or elev.zone
  
pkgs <- c("tidyverse", "hydroGOF", "readxl","zoo","tensorflow","keras","tfautograph",
          "reticulate","purrr","data.table","ggplot2","tibble","readr","plotly","caret",
          'dplyr')

inst = lapply(pkgs, library, character.only = TRUE)

# TODO:1 make 10 elevation zones as separate inputs (function to retrieve data , 1. daily, 2.3h and 3. 10 elevations)
# TODO:2 is there a better way to weigh floods more than low flows? What metrics is good for that?
# TODO:3 all (selected)-catchment runner infrastructure
# TODO:4 develop infrastructure to run scenarios with ranges of hyperparameters (units, batch-size, etc)
# TODO:5 link metrics from scenarios with field features (size, climate, elevation, etc)
# TODO:6 include n-fold replacement/solution

get_data <- function(data.dir=data.dir,fileN='2.463.3h.ptq', tstep='3h',  zones='single'){
  # 
  # pkgs <- c("tidyverse", "hydroGOF", "readxl","zoo","tensorflow","keras","tfautograph","reticulate","purrr","data.table","ggplot2","tibble","readr","plotly","caret")
  # 
  # inst = lapply(pkgs, library, character.only = TRUE)
  
  # data.dir <-file.path("Z:","GIT-repo")
  # 
  # fileN <- '2.463.24h.ptq'
  
  data <- read.table(file.path(data.dir,fileN), header=FALSE, sep=' ')
  stn_nr <- paste(strsplit(fileN, '.', fixed=TRUE)[[1]][1], strsplit(fileN, '.', fixed=TRUE)[[1]][2], sep='.')
  # ----------- Switch between 3h and 24h ----------------
  
  switch(
    tstep,
    '3h' = data[,5]<- NULL, # 3h
    '24h' = data[,4]<- 12 # 24h
  )
  
  # ------------------- pre processing  ----
  
  names(data)[1:4] <- c("year", "month", "day", "hour") # column heading columns 1:3
  
  names(data)[5:14] <- sprintf("rain%d",1:10)   # Column heading for columns 4:13
  
  names(data)[15:24] <- sprintf("temp%d",1:10)   # column headings fr columns 14:23
  
  names(data)[25] <- "flow" 
  
  
  data$dateTime = as.POSIXct(paste(paste(data$year, data$month, data$day, sep='-'), 
                                   
                                   paste(data$hour, '00', '00', sep=':')), format = '%Y-%m-%d %H:%M:%S')
  
  data[,1:4] <- NULL
  
  data[,21][data[,21] < 0] <- NA  # TODO make flow column - Make negative numbers = NA TODO 
  
  data <- data[complete.cases(data[ , 'flow']), ]   # leaves out where Q is NA
  
  d.dates <- data$dateTime
  
  # ----------- Averaging from elevations to single felt values --------------------
  
  # dat <- zoo(as.data.frame(cbind(rowMeans(data[,1:10]),rowMeans(data[11:20]), data$flow)),data$dateTime)
  # 
  # names(dat) <-c("P_mm", "T_oC", "Q_m3s")
  # 
  # 
  # data1 <- as.data.frame(as.matrix(coredata(dat)))   
  # 
  # names(data1) <-c("P_mm", "T_oC", "Q_m3s") # I don't think it's necessary
  # 
  # data1$dateTime <- d.dates
  # 
  # 
  # # functions funs() and mutate_each() will be outdated sooner or later, 
  # # need to be replaced with mutate() and across()
  # scaled_train <- data1 %>%
  #   mutate_each(funs(scale)) 
  # 
  # #TODO replace deprecated functions above
  # 
  # 
  # scaled_train <- as.data.frame(scale(data1[,-4], center = TRUE, scale = TRUE))
  # 
  # mean <- apply(data1[, -4], 2, mean,na.rm = TRUE)
  # std <- apply(data1[, -4], 2, sd, na.rm = TRUE)
  # 
  # #return(data1)
  # data1 <<-data1
  
}

#get_data(data.dir,fileN, tstep,  zones)


  # ----------- Averaging from elevations to single felt values --------------------
  
  if(zones=='single'){
    prediction <- 2
    lag <- prediction
  
  dat <- zoo(as.data.frame(cbind(rowMeans(data[,1:10]),rowMeans(data[11:20]), data$flow)),data$dateTime)
  
  names(dat) <-c("P_mm", "T_oC", "Q_m3s")
  
# For a later attempt, we should try keeping the 
# 10 elevation zones as separate input and train on those instead of 
# mean for the catchment

data1 <- as.data.frame(as.matrix(coredata(dat)))   

names(data1) <-c("P_mm", "T_oC", "Q_m3s") # I don't think it's necessary

#data2 <- data[complete.cases(data1[ , 'Q_m3s']), ]   # leaves out where Q is NA

#d.dates <- data1$dateTime

data1$dateTime <- d.dates
# 
# glimpse(data1)
# 
# #plotting 
# 
# ggplot(dat, aes(x = index(dat), y = T_oC)) + geom_line() +theme_bw()
# 
# ggplot(dat, aes(x = index(dat), y = P_mm)) + geom_line() +theme_bw()
# 
# ggplot(dat, aes(x = index(dat), y = Q_m3s)) + geom_line() +theme_bw()
# 
# 
# ggplot(dat[1:730,], aes(x = index(dat[1:730,]), y = T_oC)) + geom_line() +theme_bw()
# 
# ggplot(dat[1:730,], aes(x = index(dat[1:730,]), y = P_mm)) + geom_line() +theme_bw()
# 
# ggplot(dat[1:730,], aes(x = index(dat[1:730,]), y = `Q_m3s`)) + geom_line() +theme_bw()

# functions funs() and mutate_each() will be outdated sooner or later, 
# need to be replaced with mutate() and across()
# scaled_train <- data1 %>%
#   mutate_each(funs(scale)) 

scaled_train <- data1 %>%
  
  mutate_at(vars(),funs(scale))


scaled_train <- as.data.frame(scale(data1[,-dim(data1)[2]], center = TRUE, scale = TRUE))

scaled_train$flow <- NULL

mean <- apply(data1[, -dim(data1)[2]], 2, mean,na.rm = TRUE)

std <- apply(data1[, -dim(data1)[2]], 2, sd, na.rm = TRUE)

#glimpse(scaled_train)
#scaled_train <- scaled_train

# ---- 3D Array 
# [samples, timesteps, features] for both predict X and target Y 

# ------samples specifies the number of observations which will be processed in batches.
# ------timesteps tells us the number of time steps (lags). Or in other words how many units back in time we want our network to see.
# ------features specifies number of predictors (1 for univariate series and n for multivariate).
# In case of predictors that translates to an array of dimensions: (nrow(data) - lag - prediction + 1, 12, 1), where lag = prediction = 12.

  
  # --- lagging is done here ----

  df.lag <- shift(scaled_train[,-dim(data1)[2]], n=1:2, give.names = T)  ##column indexes of columns to be lagged as "[,startcol:endcol]", "n=1:3" sepcifies the number of lags (lag1, lag2 and lag3 in this case)
  
  x_train_data <- bind_cols(scaled_train, df.lag) # here the same amount of samples are lost as many lags are included
  
  x_train_data <- x_train_data[3:dim(x_train_data)[1],]
  
  
  #----%-wise portions for train, valid, test ----
  
  
  
}else{ # using the elevation zones 

  dat1 <- zoo(data[,-dim(data)[2]],data$dateTime)
  
  data1 <- as.data.frame(as.matrix(coredata(dat1)))   
  
  #names(data1) <-c("P_mm", "T_oC", "Q_m3s") # I don't think it's necessary
  
  data1$dateTime <- d.dates
  
  # functions funs() and mutate_each() will be outdated sooner or later, 
  # need to be replaced with mutate() and across()

  
  scaled_train <- data1 %>%
    
    mutate_at(vars(),funs(scale))
  
  #TODO replace deprecated functions above
  
  scaled_train <- as.data.frame(scale(data1[,-22], center = TRUE, scale = TRUE))
  
  mean <- apply(data1[, -22], 2, mean,na.rm = TRUE)
  std <- apply(data1[, -22], 2, sd, na.rm = TRUE)
  
  #glimpse(scaled_train)
  #scaled_train <- scaled_train
  
  # ---- 3D Array 
  # [samples, timesteps, features] for both predict X and target Y 
  
  # ------samples specifies the number of observations which will be processed in batches.
  # ------timesteps tells us the number of time steps (lags). Or in other words how many units back in time we want our network to see.
  # ------features specifies number of predictors (1 for univariate series and n for multivariate).
  # In case of predictors that translates to an array of dimensions: (nrow(data) - lag - prediction + 1, 12, 1), where lag = prediction = 12.
  
  
  # ------------- lagging is done here ----------------------
  
  df.lag <- shift(scaled_train[,1:21], n=1:2, give.names = T)  ##column indexes of columns to be lagged as "[,startcol:endcol]", "n=1:3" sepcifies the number of lags (lag1, lag2 and lag3 in this case)
  
  x_train_data <- bind_cols(scaled_train, df.lag) # here the same amount of samples are lost as many lags are included
  
  x_train_data <- x_train_data[3:dim(x_train_data)[1],]
  
  
}

# TODO  use a single input to divide data into train, validate, test

#------------- %-wise portions for train, valid, test ----

train_perc = 80 # TODO take this into function parameters

#valid_perc = 30 #% this shall be taken as a portion from "train" inside the model


train <- x_train_data[1:(100*(floor(0.01*dim(x_train_data)[1]*train_perc/100))),]  # 80% rounded down to to 100s


test <- x_train_data[(dim(train[1])[1]+1):dim(x_train_data)[1],] # rest%

#y_train_data <- train$flow 
y_train_data <- train$Q_m3s 
# y_valid_data <- valid[,3] 

y_test_data <- test$Q_m3s 

train$flow <- NULL
# valid[,3] <- NULL

test$flow <- NULL

# --------------- now transform data into 3D form ------------------------

x_train_arr <- array(
  data = as.numeric(unlist(train)),
  dim = c(
    nrow(train),
    lag,
    ncol(train)
  )
)

x_test_arr <- array(
  data = as.numeric(unlist(test)),
  dim = c(
    nrow(test),
    lag,
    ncol(test)
  )
)



y_train_arr <- array(
  data = as.numeric(unlist(y_train_data)),
  dim = c(
    length(y_train_data),
    prediction,
    1
  )
)



y_test_arr <- array(
  data = as.numeric(unlist(y_test_data)),
  dim = c(
    length(y_test_data),
    prediction,
    1
  )
)

# --------------- pre parameter seeting -------------------

# TODO add to this function parameters

batch_size1 = 200 #200, 300, 400, 500) #i 10 /20
time_step1 = 10 #k 2 - 100
units1 = 128 #j 128
epochs1 = 100 #

# batch_size1 = 100 #200, 300, 400, 500) #i
# time_step1 = 10 #k
# units1 = 128 #j
# epochs1 = 20 #
# prepare input data for the prediction

# kfold <- createFolds(x_train_data, k=10)
# 
# for (fold in kfold){

# define 10-fold cross validation

# running through each fold of the cross-validation

#  ------------------ Set up ML LSTM model -------------------

lstm_model <- keras_model_sequential()

lstm_model %>%
  layer_lstm(units = units1, # size of the layer (was 64)
             activation = 'tanh',
             batch_input_shape = c(batch_size1, time_step1, 62), # batch size, timesteps, features
             kernel_regularizer = regularizer_l2(0.001),
             return_sequences = TRUE,
             stateful = TRUE) %>%
  #layer_batch_normalization()%>%
  # fraction of the units to drop for the linear transformation of the inputs
  layer_dropout(rate = 0.2) %>%
 
   layer_lstm(units = units1/2,
             activation = 'sigmoid',
             kernel_regularizer = regularizer_l2(0.001),
             return_sequences = TRUE,
             stateful = TRUE) %>%
  layer_dropout(rate = 0.2) %>%
  #layer_batch_normalization()%>%
  time_distributed(keras::layer_dense(units = 1))

# Compile the model

lstm_model %>%
  
  # compile(loss = 'mae', optimizer = 'adam', metrics = 'accuracy')
  compile(loss = 'mae', optimizer = optimizer_adam())

summary(lstm_model)

history2 <- lstm_model %>% fit(
  x = x_train_arr,
  y = y_train_arr,
  batch_size = batch_size1, # = batch_size in line 226
  epochs = epochs1, # was 100
  validation_split=0.2,
  # callbacks = c(
  #   callback_early_stopping(patience = 5),
  #   callback_reduce_lr_on_plateau(factor = 0.05)
  # ),
  verbose = 0,
  shuffle = FALSE
)

# ----------------end of training ML model -----------------

plot(history2) +theme_bw()

#}

lstm_train <- lstm_model %>%
  predict(x_train_arr, batch_size = batch_size1) %>%   
  .[, , 1]

lstm_forecast <- lstm_model %>%
  predict(x_test_arr[1:5500,,], batch_size = batch_size1) %>%   #5500 must be multile of batch size
  .[, , 1]

lstm_test <- as.data.frame(cbind(d.dates[(dim(train)[1]+1):(dim(train)[1]+dim(lstm_forecast)[1])],
                                 as.data.frame(y_test_data[1:dim(lstm_forecast)[1]]),
                                 as.data.frame(lstm_forecast)))

lstm_train1 <- as.data.frame(cbind(d.dates[1:dim(train)[1]],
                                   as.data.frame(y_train_data[1:dim(lstm_train)[1]]),
                                   as.data.frame(lstm_train)))

names(lstm_train1) <- c('dates','obs1', 'sim1','sim2')
names(lstm_test) <- c('dates','obs1', 'sim1','sim2')
# 

usc_lstm_train <- as.data.frame(cbind(dates=lstm_train1$dates, std[3]*lstm_train1[,2:4]+(mean[3])))
usc_lstm_test <- as.data.frame(cbind(dates=lstm_test$dates, std[3]*lstm_test[,2:4]+(mean[3])))

##------- export aes from trainings -------------

#write.csv(resultData, file.path(data.dir,'Qtd-1-9-train-100epochs.csv'))
#write.xlsx(resultData, file.path(data.dir,'Qtd-1-9-sel-train-200epochs.xlsx'))


#--- end function

#-------Collect last values and means of MAE from history2-metrics-loss

plot(history2)+theme_bw()
# perform the prediction:

# check the validation data


plot_ly(usc_lstm_test, x = ~dates, y = ~obs1, type = "scatter", mode = "lines")%>% #, color = ~T_oC) %>%
  add_trace(y = usc_lstm_test$sim1, x = ~dates, name = "LSTM prediction", mode = "lines",color="brown")
plot_ly(usc_lstm_train, x = ~dates, y = ~obs1, type = "scatter", mode = "lines")%>% #, color = ~T_oC) %>%
  add_trace(y = usc_lstm_train$sim1, x = ~dates, name = "LSTM prediction", mode = "lines",color="brown")

# TODO: make this into a table for better handling, perhaps add composite unit error (or sg like that)
# rmse_trn <- rmse(usc_lstm_train$sim1, usc_lstm_train$obs1)
# kge_trn <- KGE(usc_lstm_train$sim1, usc_lstm_train$obs1)
# nse_trn <- NSE(usc_lstm_train$sim1, usc_lstm_train$obs1)

eval_tbl1 <- as_tibble(stn=stn_nr, rmse_tst = rmse(usc_lstm_test$sim1, usc_lstm_test$obs1),
                        kge_tst = KGE(usc_lstm_test$sim1, usc_lstm_test$obs1),
                        nse_tst = NSE(usc_lstm_test$sim1, usc_lstm_test$obs1),
                        ipe_tst = (rmse_tst^2+kge_tst^2+nse_tst^2)^.5)

eval_tbl <-bind_rows(eval_tbl, eval_tbl1) #make sure it is defined outside the loop

## Warning: line.color doesn't (yet) support data arrays
plot_ly(dat.df[12001:18000,], x = ~index, y = ~Q_m3s, type = "scatter", mode = "lines") %>%
  add_trace(y = pred_out, x = dat.df$index[12001:18000], name = "LSTM prediction", mode = "lines")


}


# hyetograph
