

#İmporting libraries
library(readxl)
library(tidyverse)
library(ggplot2)
library(ggmosaic)
library(tm)
library(SnowballC)
library(NLP)
library(e1071)
library(kernlab)
library(caTools)
library(randomForest)
library(dplyr)
library(caret)


#Importing data
fake = read.csv("Fake.csv")
true = read.csv("True.csv")

#$Data Preprocessing
fake$class = 0
true$class = 1
dataset = rbind(fake,true)
dataset$date = gsub(",", "",dataset$date)
#Date should be converted to date format
dataset$date = gsub("December","12",dataset$date)
dataset$date = gsub("january","01",dataset$date)
dataset$date = gsub("February","02",dataset$date)
dataset$date = gsub("March","03",dataset$date)
dataset$date = gsub("April","04",dataset$date)
dataset$date = gsub("May","05",dataset$date)
dataset$date = gsub("June","06",dataset$date)
dataset$date = gsub("July","07",dataset$date)
dataset$date = gsub("August","08",dataset$date)
dataset$date = gsub("September","09",dataset$date)
dataset$date = gsub("October","10",dataset$date)
dataset$date = gsub("November","11",dataset$date)

dataset$date = gsub(" ","-",dataset$date)
dataset$date <- as.Date(dataset$date,format="%m-%d-%Y")
dataset %>%
  summarise_all(funs(sum(is.na(.))))
dataset$title = as.character(dataset$title)
dataset$text = as.character(dataset$text)
dataset$class = factor(dataset$class)


#Text Cleaning for title column 
corpus = VCorpus(VectorSource(dataset$title))
corpus = tm_map(corpus, content_transformer(tolower))
corpus = tm_map(corpus, removeNumbers)
corpus = tm_map(corpus, removePunctuation)
corpus = tm_map(corpus, removeWords, stopwords())
corpus = tm_map(corpus, stemDocument)
corpus = tm_map(corpus, stripWhitespace)
                            
#Creating the Bag of Words Model for title column 
dtm_title = DocumentTermMatrix(corpus)
dtm_title = removeSparseTerms(dtm_title, 0.999)
dataset_title = as.data.frame(as.matrix(dtm_title))
dataset_title$classFakeTrue = dataset$class

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
set.seed(123)
split = sample.split(dataset_title$classFakeTrue, SplitRatio = 0.8)
training_set = subset(dataset_title, split == TRUE)
test_set = subset(dataset_title, split == FALSE)


# Fitting Random Forest Classification to the Training set
# install.packages('randomForest')
set.seed(123)
classifier = randomForest(x = training_set[-1615],
                          y = training_set$classFakeTrue,
                          ntree = 10)

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-1615])

# Making the Confusion Matrix
cm = table(test_set[, 1615], y_pred)

y_pred_title <- as.data.frame(y_pred)
cm_title <- confusionMatrix(data = y_pred, reference = test_set$classFakeTrue)
draw_confusion_matrix(cm_title)

#Ploting confusion matrix(function is from stackoverflow.com/questions/23891140/r-how-to-visualize-confusion-matrix-using-the-caret-package) 
draw_confusion_matrix <- function(cm) {
  
  total <- sum(cm$table)
  res <- as.numeric(cm$table)
  
  # Generate color gradients. Palettes come from RColorBrewer.
  greenPalette <- c("#F7FCF5","#E5F5E0","#C7E9C0","#A1D99B","#74C476","#41AB5D","#238B45","#006D2C","#00441B")
  redPalette <- c("#FFF5F0","#FEE0D2","#FCBBA1","#FC9272","#FB6A4A","#EF3B2C","#CB181D","#A50F15","#67000D")
  getColor <- function (greenOrRed = "green", amount = 0) {
    if (amount == 0)
      return("#FFFFFF")
    palette <- greenPalette
    if (greenOrRed == "red")
      palette <- redPalette
    colorRampPalette(palette)(100)[10 + ceiling(90 * amount / total)]
  }
  
  # set the basic layout
  layout(matrix(c(1,1,2)))
  par(mar=c(2,2,2,2))
  plot(c(100, 345), c(300, 450), type = "n", xlab="", ylab="", xaxt='n', yaxt='n')
  title('CONFUSION MATRIX', cex.main=2)
  
  # create the matrix 
  classes = colnames(cm$table)
  rect(150, 430, 240, 370, col=getColor("green", res[1]))
  text(195, 435, classes[1], cex=1.2)
  rect(250, 430, 340, 370, col=getColor("red", res[3]))
  text(295, 435, classes[2], cex=1.2)
  text(125, 370, 'Predicted', cex=1.3, srt=90, font=2)
  text(245, 450, 'Actual', cex=1.3, font=2)
  rect(150, 305, 240, 365, col=getColor("red", res[2]))
  rect(250, 305, 340, 365, col=getColor("green", res[4]))
  text(140, 400, classes[1], cex=1.2, srt=90)
  text(140, 335, classes[2], cex=1.2, srt=90)
  
  # add in the cm results
  text(195, 400, res[1], cex=1.6, font=2, col='white')
  text(195, 335, res[2], cex=1.6, font=2, col='white')
  text(295, 400, res[3], cex=1.6, font=2, col='white')
  text(295, 335, res[4], cex=1.6, font=2, col='white')
  
  # add in the specifics 
  plot(c(100, 0), c(100, 0), type = "n", xlab="", ylab="", main = "DETAILS", xaxt='n', yaxt='n')
  text(10, 85, names(cm$byClass[1]), cex=1.2, font=2)
  text(10, 70, round(as.numeric(cm$byClass[1]), 3), cex=1.2)
  text(30, 85, names(cm$byClass[2]), cex=1.2, font=2)
  text(30, 70, round(as.numeric(cm$byClass[2]), 3), cex=1.2)
  text(50, 85, names(cm$byClass[5]), cex=1.2, font=2)
  text(50, 70, round(as.numeric(cm$byClass[5]), 3), cex=1.2)
  text(70, 85, names(cm$byClass[6]), cex=1.2, font=2)
  text(70, 70, round(as.numeric(cm$byClass[6]), 3), cex=1.2)
  text(90, 85, names(cm$byClass[7]), cex=1.2, font=2)
  text(90, 70, round(as.numeric(cm$byClass[7]), 3), cex=1.2)
  
  # add in the accuracy information 
  text(30, 35, names(cm$overall[1]), cex=1.5, font=2)
  text(30, 20, round(as.numeric(cm$overall[1]), 3), cex=1.4)
  text(70, 35, names(cm$overall[2]), cex=1.5, font=2)
  text(70, 20, round(as.numeric(cm$overall[2]), 3), cex=1.4)
} 

#Visulization of simple dataset
#Plotting distribution of fake and real new in time 
dataset_plot_date_dif = dataset %>%
  group_by(date, class) %>%
  tally()
ggplot(dataset_plot_date_dif, aes(x = date, y = n)) + 
  geom_point(aes(size = class), alpha = 0.25,color = "azure4") + 
  coord_cartesian(ylim = c(0, 50)) + 
  geom_smooth( aes(color = class)) + 
  guides(color=guide_legend(title="Fake and Real Smooth", )) +
  guides(size=guide_legend(title="Fake and Real Point")) +
  ggtitle("Changes in the Number of Real-Fake News in Years") +
  xlab("Year") +
  ylab("Total Number of News") +
  theme(
    plot.title = element_text(color="darkslategray4", size=14, face="bold.italic", hjust = 0.5),
    axis.title.x = element_text(color="darkslategray4", size=14, face="bold.italic", hjust = 0.5),
    axis.title.y = element_text(color="darkslategray4", size=14, face="bold.italic", hjust = 0.5),
    axis.text.x = element_text(color="bisque3", size=14, face="bold.italic", hjust = 0.5, angle = 15),
    axis.text.y = element_text(color="bisque3", size=14, face="bold.italic", hjust = 0.5),
    legend.position = c(0.13,0.8),
    legend.background = element_rect(fill = "darkgray"),
  ) +
  scale_size_discrete(name="Fake and Real Point",
                        breaks=c("0", "1"),
                        labels=c("Fake", "Real")) + 
  scale_color_discrete(name="Fake and Real Smooth",
                      breaks=c("0", "1"),
                      labels=c("Fake", "Real"))
  


#----------------------Text model with naive bayes model

#Text Cleaning for text column 
corpus_text = VCorpus(VectorSource(dataset$text))
corpus_text = tm_map(corpus_text, content_transformer(tolower))
corpus_text = tm_map(corpus_text, removeNumbers)
corpus_text = tm_map(corpus_text, removePunctuation)
corpus_text = tm_map(corpus_text, removeWords, stopwords())
corpus_text = tm_map(corpus_text, stemDocument)
corpus_text = tm_map(corpus_text, stripWhitespace)

#Creating the Bag of Words Model for text column 
dtm_text = DocumentTermMatrix(corpus_text)
dtm_text = removeSparseTerms(dtm_text, 0.90)
dataset_text = as.data.frame(as.matrix(dtm_text))

#Converting multiple words to yes or no version
convert_count <- function(x) {
  y <- ifelse(x > 0, 1,0)
  y <- factor(y, levels=c(0,1), labels=c("0", "1"))
  y
}
dataset_text_analysis <- apply(dataset_text, 2, convert_count)
dataset_text_analysis = as.data.frame(as.matrix(dataset_text_analysis))
dataset_text_analysis$classFakeTrue = dataset$class

# Splitting the dataset into the Training set and Test set
set.seed(123)
split_text = sample.split(dataset_text_analysis$classFakeTrue, SplitRatio = 0.8)
training_set_text = subset(dataset_text_analysis, split_text == TRUE)
test_set_text = subset(dataset_text_analysis, split_text == FALSE)

# Fitting Naive Bayes to the Training set
# install.packages('e1071')
library(e1071)
classifier_text = naiveBayes(x = training_set_text[-278],
                        y = training_set$classFakeTrue)

# Predicting the Test set results
y_pred_text = predict(classifier_text, newdata = test_set_text[-278])

# Making the Confusion Matrix
cm_text = table(test_set_text[, 278], y_pred_text)


#Ploting confusion matrix(function is from stackoverflow.com/questions/23891140/r-how-to-visualize-confusion-matrix-using-the-caret-package) 
y_pred_text2 <- as.data.frame(y_pred_text)
cm2 <- confusionMatrix(data = y_pred_text, reference = test_set_text$classFakeTrue)
draw_confusion_matrix(cm2)

