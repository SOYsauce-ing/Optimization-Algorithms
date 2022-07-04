#image process
install.packages("BiocManager")
BiocManager::install("EBImage")

library(EBImage)
#read concept
library(readr)
TreeLabels <- read.csv("ProjectData/TreeLabels.csv")
TreeLabels <- TreeLabels[,-c(2:8)]

colnames(TreeLabels)[2] <- "TreeLabels"
colnames(TreeLabels)[1] <- "Image"
AnimalLabels <- read.csv("ProjectData/AnimalLabels.csv")
AnimalLabels <- AnimalLabels[,-c(2:8)]
colnames(AnimalLabels)[1] <- "Image"
colnames(AnimalLabels)[2] <- "AnimalLabels"
MythologicalLabels <- read.csv("ProjectData/MythologicalLabels.csv")
MythologicalLabels <- MythologicalLabels[,-c(2:8)]
colnames(MythologicalLabels)[2] <- "MythologicalLabels_1"
colnames(MythologicalLabels)[3] <- "MythologicalLabels_2"
colnames(MythologicalLabels)[1] <- "Image"
concept <- merge(TreeLabels,AnimalLabels)
concept <- merge(concept,MythologicalLabels)
rm(TreeLabels,AnimalLabels,MythologicalLabels)


#read images
list<-list.files(path = "ProjectData/s1/img/", pattern = ".jpg",full.names = TRUE)
list<-append(list,list.files(path = "ProjectData/s2/img/", pattern = ".jpg",full.names = TRUE))
list<-append(list,list.files(path = "ProjectData/s3/img/", pattern = ".jpg",full.names = TRUE))
list<-append(list,list.files(path = "ProjectData/s4/img/", pattern = ".jpg",full.names = TRUE))
list<-append(list,list.files(path = "ProjectData/s5/img/", pattern = ".jpg",full.names = TRUE))
list<-append(list,list.files(path = "ProjectData/s6/img/", pattern = ".jpg",full.names = TRUE))
list<-append(list,list.files(path = "ProjectData/s7/img/", pattern = ".jpg",full.names = TRUE))
list<-append(list,list.files(path = "ProjectData/s8/img/", pattern = ".jpg",full.names = TRUE))
list<-append(list,list.files(path = "ProjectData/s9/img/", pattern = ".jpg",full.names = TRUE))
list<-append(list,list.files(path = "ProjectData/s10/img/", pattern = ".jpg",full.names = TRUE))
list<-append(list,list.files(path = "ProjectData/s11/img/", pattern = ".jpg",full.names = TRUE))
list<-append(list,list.files(path = "ProjectData/s12/img/", pattern = ".jpg",full.names = TRUE))
list<-append(list,list.files(path = "ProjectData/s13/img/", pattern = ".jpg",full.names = TRUE))
list_of_images <- lapply(list,readImage)
images <- lapply(list_of_images, resize,w=10,h=10)
rm(list_of_images,list)


#flatten image data
image_df <- data.frame(matrix(ncol = 12))
image_df <- image_df[-1,]
for (i in 1:106){
  df <- as.vector(images[[i]])
  image_df <- rbind(image_df,df)
}
rm(images,df)

for (i in 1:ncol(image_df)){
  colnames(image_df)[i] <- paste0("cube", i)
}



#Text Data
library(tm)
library(dplyr)
library(stringr)

removeCommonTerms <- function(X, percentage) {
  x = t(X)
  t = table(x$i) < x$ncol * percentage
  X[, as.numeric(names(t[t]))]
}

corpus = Corpus(DirSource("ProjectData/text data", encoding = "UTF-8", recursive = TRUE, ignore.case = FALSE, mode = "text"),
) %>%
  tm_map(content_transformer(tolower)) %>%        # no uppercase
  tm_map(removeWords, stopwords('en')) %>%        # remove stopwords
  tm_map(removePunctuation) %>%                   # no punctuation
  tm_map(stripWhitespace) %>%                     # no extra whitespaces
  tm_map(stemDocument) %>%                        # reduce to radical
  DocumentTermMatrix(
    control = list(
      weighting = weightTf,
      wordLengths = c(3,30),                  # radical between 3 and 30
      minDocFreq = 1                          # appears at least 1 times
    )
    
  ) %>%
  removeCommonTerms(0.70) %>%                     # maximum 70% documents
  as.matrix()

textdata <- data.frame(corpus, concept[,-1])
rownames(textdata) <- NULL
text_df <- textdata[lapply(textdata, function(x) sum(x==0) / length(x) ) < 0.95]
rm(textdata)

#Image and text data
ImageText_df <- cbind(image_df, text_df)
rm(corpus,i)


#Image and text by feature
Tree_df <- varianceScaling(ImageText_df[, 1:459])
Animal_df <- varianceScaling(ImageText_df[, c(1:458,460)])
Myth1_df <- varianceScaling(ImageText_df[, c(1:458, 461)])
Myth2_df <- varianceScaling(ImageText_df[, c(1:458, 462)])


#ADAGRAD
#Tree Labels 
library(gradDescent)
data <- splitData(Tree_df$scaledDataSet, dataTrainRate = 0.7)
model <- ADAGRAD(data$dataTrain, alpha=0.01, maxIter=300, seed=NULL)
Test_set <- data$dataTest[,-c(459)]
prediction <- prediction(model,Test_set)
error <- (data$dataTest)[459] - prediction[459]
MSE_T_ADA <- sum(error ** 2) / length(error)


#Animal Labels
data <- splitData(Animal_df$scaledDataSet, dataTrainRate = 0.7)
model <- ADAGRAD(data$dataTrain, alpha=0.01, maxIter=600, seed=NULL)
Test_set <- data$dataTest[,-c(459)]
prediction <- prediction(model,Test_set)
error <- (data$dataTest)[459] - prediction[459]
MSE_A_ADA <- sum(error ** 2) / length(error)


#Myth1 Labels
data <- splitData(Myth1_df$scaledDataSet, dataTrainRate = 0.7)
model <- ADAGRAD(data$dataTrain, alpha=0.01, maxIter=500, seed=NULL)
Test_set <- data$dataTest[,-c(459)]
prediction <- prediction(model,Test_set)
error <- (data$dataTest)[459] - prediction[459]
MSE_M1_ADA <- sum(error ** 2) / length(error)


#Myth2 Labels
data <- splitData(Myth2_df$scaledDataSet, dataTrainRate = 0.7)
model <- ADAGRAD(data$dataTrain, alpha=0.01, maxIter=500, seed=NULL)
Test_set <- data$dataTest[,-c(459)]
prediction <- prediction(model,Test_set)
error <- (data$dataTest)[459] - prediction[459]
MSE_M2_ADA <- sum(error ** 2) / length(error)



#MGD
#Tree Labels
data <- splitData(Tree_df$scaledDataSet, dataTrainRate = 0.7)
model <- MGD(data$dataTrain, alpha=0.01, maxIter=500, momentum = 0.9)
Test_set <- data$dataTest[,-c(459)]
prediction <- prediction(model,Test_set)
error <- (data$dataTest)[459] - prediction[459]
MSE_T_MGD <- sum(error ** 2) / length(error)


#Animal Labels
data <- splitData(Animal_df$scaledDataSet, dataTrainRate = 0.7)
model <- MGD(data$dataTrain, alpha=0.01, maxIter=500, momentum = 0.9)
Test_set <- data$dataTest[,-c(459)]
prediction <- prediction(model,Test_set)
error <- (data$dataTest)[459] - prediction[459]
MSE_A_MGD <- sum(error ** 2) / length(error)


#Myth1 Labels
data <- splitData(Myth1_df$scaledDataSet, dataTrainRate = 0.7)
model <- MGD(data$dataTrain, alpha=0.01, maxIter=500, momentum = 0.9)
Test_set <- data$dataTest[,-c(459)]
prediction <- prediction(model,Test_set)
error <- (data$dataTest)[459] - prediction[459]
MSE_M1_MGD <- sum(error ** 2) / length(error)


#Myth2 Labels
data <- splitData(Myth2_df$scaledDataSet, dataTrainRate = 0.7)
model <- MGD(data$dataTrain, alpha=0.01, maxIter=500, momentum = 0.9)
Test_set <- data$dataTest[,-c(459)]
prediction <- prediction(model,Test_set)
error <- (data$dataTest)[459] - prediction[459]
MSE_M2_MGD <- sum(error ** 2) / length(error)





