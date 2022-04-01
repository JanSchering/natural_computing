library(pROC)

working_directory <- getwd()

data_directory <- file.path(getwd(), "test")
scores_directory <- file.path(data_directory, "processed")

# ----------------------------------- Cert1 ------------------------------------------------
cert_1 <- read.csv(file.path(scores_directory, "cert1r1.csv"), sep=",", header=TRUE)
cert_1 <- cert_1[order(cert_1$scores),]
cert_1_roc <- roc(response = cert_1$label, predictor = cert_1$anomaly_score)
plot(cert_1_roc)
print(cert_1_roc$auc)

cert_1 <- read.csv(file.path(scores_directory, "cert1r2.csv"), sep=",", header=TRUE)
cert_1 <- cert_1[order(cert_1$scores),]
cert_1_roc <- roc(response = cert_1$label, predictor = cert_1$anomaly_score)
plot(cert_1_roc, add = TRUE, col = "red")
print(cert_1_roc$auc)

cert_1 <- read.csv(file.path(scores_directory, "cert1r3.csv"), sep=",", header=TRUE)
cert_1 <- cert_1[order(cert_1$scores),]
cert_1_roc <- roc(response = cert_1$label, predictor = cert_1$anomaly_score)
plot(cert_1_roc, cex.lab=2, cex.axis=2, cex.main=2, cex.sub=2)
print(cert_1_roc$auc)

cert_1 <- read.csv(file.path(scores_directory, "cert1.csv"), sep=",", header=TRUE)
cert_1 <- cert_1[order(cert_1$scores),]
cert_1_roc <- roc(response = cert_1$label, predictor = cert_1$anomaly_avg)
plot(cert_1_roc, add = TRUE, col = "red")
print(cert_1_roc$auc)

legend("bottomright", legend=c("r=3", "r=4"),
       col=c(par("fg"), "red"), lwd=2, cex=2)

cert_1 <- read.csv(file.path(scores_directory, "cert1r5.csv"), sep=",", header=TRUE)
cert_1 <- cert_1[order(cert_1$scores),]
cert_1_roc <- roc(response = cert_1$label, predictor = cert_1$anomaly_score)
plot(cert_1_roc, add = TRUE, col = "orange", lty=4)
print(cert_1_roc$auc)

cert_1 <- read.csv(file.path(scores_directory, "cert1r6.csv"), sep=",", header=TRUE)
cert_1 <- cert_1[order(cert_1$scores),]
cert_1_roc <- roc(response = cert_1$label, predictor = cert_1$anomaly_score)
plot(cert_1_roc, add = TRUE, col = "purple", lty=3, pch="o")
print(cert_1_roc$auc)

cert_1 <- read.csv(file.path(scores_directory, "cert1r7.csv"), sep=",", header=TRUE)
cert_1 <- cert_1[order(cert_1$scores),]
cert_1_roc <- roc(response = cert_1$label, predictor = cert_1$anomaly_score)
plot(cert_1_roc, add = TRUE, col = "dodgerblue2", lty=2)
print(cert_1_roc$auc)

legend("bottomright", legend=c("r=1", "r=2", "r=3", "r=4", "r=5", "r=6", "r=7"),
       col=c(par("fg"), "red", "green", "midnightblue", "orange", "purple", "dodgerblue2"), lwd=2)

# ------------------------------- Cert2 ---------------------------------------
cert_2 <- read.csv(file.path(scores_directory, "cert2r3.csv"), sep=",", header=TRUE)
cert_2 <- cert_2[order(cert_2$scores),]
cert_2_roc <- roc(response = cert_2$label, predictor = cert_2$anomaly_score)
plot(cert_2_roc, cex.lab=2, cex.axis=2, cex.main=2, cex.sub=2)
print(cert_2_roc$auc)

cert_2 <- read.csv(file.path(scores_directory, "cert2.csv"), sep=",", header=TRUE)
cert_2 <- cert_2[order(cert_2$scores),]
cert_2_roc <- roc(response = cert_2$label, predictor = cert_2$anomaly_avg)
plot(cert_2_roc, add = TRUE, col = "red")
print(cert_2_roc$auc)

legend("bottomright", legend=c("r=3", "r=4"),
       col=c(par("fg"), "red"), lwd=2, cex=2)

# ------------------------------- Cert3 ----------------------------------------
cert_3 <- read.csv(file.path(scores_directory, "cert3r3.csv"), sep=",", header=TRUE)
cert_3 <- cert_3[order(cert_3$scores),]
cert_3_roc <- roc(response = cert_3$label, predictor = cert_3$anomaly_score)
plot(cert_3_roc, cex.lab=2, cex.axis=2, cex.main=2, cex.sub=2)
print(cert_3_roc$auc)

cert_3 <- read.csv(file.path(scores_directory, "cert3.csv"), sep=",", header=TRUE)
cert_3 <- cert_3[order(cert_3$scores),]
cert_3_roc <- roc(response = cert_3$label, predictor = cert_3$anomaly_avg)
plot(cert_3_roc, add = TRUE, col = "red", lty=2)
print(cert_3_roc$auc)

legend("bottomright", legend=c("r=3", "r=4"),
       col=c(par("fg"), "red"), lwd=2, cex=2)

# ------------------------------ Unm1 -----------------------------------------
unm_1 <- read.csv(file.path(scores_directory, "unm1r3.csv"), sep=",", header=TRUE)
unm_1 <- unm_1[order(unm_1$scores),]
unm_1_roc <- roc(response = unm_1$label, predictor = unm_1$anomaly_score)
plot(unm_1_roc, , cex.lab=2, cex.axis=2, cex.main=2, cex.sub=2)
print(unm_1_roc$auc)

unm_1 <- read.csv(file.path(scores_directory, "unm1.csv"), sep=",", header=TRUE)
unm_1 <- unm_1[order(unm_1$scores),]
unm_1_roc <- roc(response = unm_1$label, predictor = unm_1$anomaly_score)
plot(unm_1_roc, add = TRUE, col = "red")
print(unm_1_roc$auc)

legend("bottomright", legend=c("r=3", "r=4"),
       col=c(par("fg"), "red"), lwd=2, cex=2)

# ---------------------------- Unm2 ------------------------------------------
unm_2 <- read.csv(file.path(scores_directory, "unm2r3.csv"), sep=",", header=TRUE)
unm_2 <- unm_2[order(unm_2$scores),]
unm_2_roc <- roc(response = unm_2$label, predictor = unm_2$anomaly_score)
plot(unm_2_roc, cex.lab=2, cex.axis=2, cex.main=2, cex.sub=2)
print(unm_2_roc$auc)

unm_2 <- read.csv(file.path(scores_directory, "unm2.csv"), sep=",", header=TRUE)
unm_2 <- unm_2[order(unm_2$scores),]
unm_2_roc <- roc(response = unm_2$label, predictor = unm_2$anomaly_score)
plot(unm_2_roc, add = TRUE, col = "red")
print(unm_2_roc$auc)

legend("bottomright", legend=c("r=3", "r=4"),
       col=c(par("fg"), "red"), lwd=2, cex=2)

# -------------------------- Unm3 -------------------------------------------
unm_3 <- read.csv(file.path(scores_directory, "unm3r3.csv"), sep=",", header=TRUE)
unm_3 <- unm_3[order(unm_3$scores),]
unm_3_roc <- roc(response = unm_3$label, predictor = unm_3$anomaly_score)
plot(unm_3_roc, cex.lab=2, cex.axis=2, cex.main=2, cex.sub=2)
print(unm_3_roc$auc)

unm_3 <- read.csv(file.path(scores_directory, "unm3.csv"), sep=",", header=TRUE)
unm_3 <- unm_3[order(unm_3$scores),]
unm_3_roc <- roc(response = unm_3$label, predictor = unm_3$anomaly_score)
plot(unm_3_roc, add = TRUE, col = "red")
print(unm_3_roc$auc)

legend("bottomright", legend=c("r=3", "r=4"),
       col=c(par("fg"), "red"), lwd=2, cex=2)




