library(pROC)

working_directory <- getwd()

data_directory <- file.path(getwd(), "Archive")

data_eng <- read.csv(file.path(data_directory, "english.test"), sep="\n", header=FALSE)
data_eng$label <- 0

data_tag <- read.csv(file.path(data_directory, "tagalog.test"), sep="\n", header=FALSE)
data_tag$label <- 1

data_hili <- read.csv(file.path(data_directory, "hiligaynon.txt"), sep="\n", header=FALSE)
data_hili$label <- 1

data_middle <- read.csv(file.path(data_directory, "middle-english.txt"), sep="\n", header=FALSE)
data_middle$label <- 1

data_plaut <- read.csv(file.path(data_directory, "plautdietsch.txt"), sep="\n", header=FALSE)
data_plaut$label <- 1

data_xhosa <- read.csv(file.path(data_directory, "xhosa.txt"), sep="\n", header=FALSE)
data_xhosa$label <- 1



scores <- read.csv(file.path(data_directory, "anomaly_score_plautdietsch_r_1.txt"), sep="\n", header=FALSE)
scores <- unlist(scores)

data <- rbind(data_eng, data_plaut)
data$score <- scores
data <- data[order(data$score),]

data_roc <- roc(response = data$label, predictor = data$score)

plot(data_roc)

print(data_roc$auc)

