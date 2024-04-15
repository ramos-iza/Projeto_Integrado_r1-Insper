library(tidyverse)
library(ggplot2)
library(glmnet)
library(Metrics)
library(pROC)
library(PerformanceAnalytics)
library(readxl)
library(tibble)
library(rsample)
library(rpart)
library(partykit)
library(ISLR)
library(vip)

# Importando os dados

df <- read.csv('/Users/izadoraramos/Desktop/Insper/ProjetoIntegrador/df_entrega.csv') 
df

str(df)

# Retirando colunas comp_id ----------------------------------------------------
colunas_para_remover <- c("comp_id", "sales")
df <- subset(df, select = -which(names(df) %in% colunas_para_remover))
df

# plotando para ver a distribuição -----------------------------------------------
ggplot(df, aes(x = no_operating)) +
  geom_histogram(fill = "blue") +
  labs(title = "Distibuição das operações das empresas")
theme_minimal()

# Separando base em treino e teste  ---------------------------------------

set.seed(42)
partition <- sample(nrow(df), size = .70 * nrow(df), replace = FALSE)

# Transformando valores de treinamento e teste em matriz para a função glmnet --------------------------------------------------
df_tr <- df[partition, ]
x_tr <- model.matrix(~ . -no_operating, df_tr)
print(nrow(x_tr))
y_tr <- df$no_operating[partition]
print(length(y_tr))

df_test <- df[-partition, ]
x_test <- model.matrix(~ . - no_operating, df_test)
print(nrow(x_test))
y_test <- df$no_operating[-partition]
print(length(y_test))

# Data frame resultados  --------------------------------------------------

resultados <- tibble(modelo = c("logistica", "ridge", "lasso", "elastic_net", "arvore_decisao"), 
                     acuracia = NA, 
                     auc = NA)

# Regressão Logística  ----------------------------------------------------

# Inicializar um dataframe para armazenar os resultados de corte e acurácia
df_cortes_acuracia <- data.frame(Corte=numeric(), Acuracia=numeric())

# Criar um vetor de cortes de 0.1 até 1, com incremento de 0.01
cortes <- seq(0.1, 1, by=0.01)

# Dataframe para comparar cortes e acurácia 
df_cortes_acuracia <- data.frame(Corte=numeric(length(cortes)), Acuracia=numeric(length(cortes)))

# Treinando com modelo logístico
modelo_logistico <- glm(no_operating ~ ., df_tr, family = "binomial")

# Aplicando o modelo na base de teste
prob_logistica <- predict(modelo_logistico, df_test, type = "response")

# Encontrando o corte com a melhor acurácia
for(i in 1:length(cortes)){
  corte_atual <- cortes[i]
  
  acuracia_atual <- mean(y_test == ifelse(prob_logistica >= corte_atual, 1, 0))
  
  df_cortes_acuracia$Corte[i] <- corte_atual
  df_cortes_acuracia$Acuracia[i] <- acuracia_atual
}

# Exibir os resultados
print(df_cortes_acuracia)

# Selecionar o corte com a maior acurácia
melhor_corte <- df_cortes_acuracia[which.max(df_cortes_acuracia$Acuracia), ]
print(paste("Melhor corte: ", melhor_corte$Corte, " com acurácia: ", melhor_corte$Acuracia, sep=""))

corte <- melhor_corte$Corte

# Plotando a área sob a curva
roc_logistica <- roc(y_test, prob_logistica)
plot(roc_logistica, main="Curva ROC", col="#1c61b6")

# Calcular a acurácia e a curva ROC com o melhor corte

resultados$acuracia[resultados$modelo == "logistica"] <- mean(y_test == ifelse(prob_logistica >= corte, 1, 0))

resultados$auc[resultados$modelo == "logistica"] <- roc(y_test, prob_logistica)$auc

resultados

vip::vip(modelo_logistico, aesthetics = list(fill = "#FF5757"))


# ------------------------------------------------------------------------------------------

# Regressão Logística com método de encolhimento Ridge ----------------------------------------------------


# Inicializar um dataframe para armazenar os resultados de corte e acurácia
df_cortes_acuracia_ridge <- data.frame(Corte=numeric(), Acuracia=numeric())

# Criar um vetor de cortes de 0.1 até 1, com incremento de 0.01
cortes <- seq(0.1, 1, by=0.01)

# Dataframe para comparar cortes e acurácia 
df_cortes_acuracia_ridge <- data.frame(Corte=numeric(length(cortes)), Acuracia=numeric(length(cortes)))

# Treinando o modelo ridge
modelo_ridge <- glmnet(x = x_tr, y = y_tr, family = "binomial", alpha = 0)

# ridge - validação cruzada -----------------------------------------------

cv_ridge <- cv.glmnet(x = x_tr, y = y_tr, alpha = 0)

summary(cv_ridge)

plot(cv_ridge, cex.lab = 1.3)

# Aplicando a base de teste com o lambda ótimo 

prob_ridge <- predict(modelo_ridge, newx = x_test, s = cv_ridge$lambda.1se, type = "response") # valor predito
colnames(prob_ridge)

# Transformando a matriz de probabilidade em vetor para calcular a área sob a curva (ROC)
prob_ridge <- as.numeric(prob_ridge[, 1])

# Encontrando o corte com a melhor acurácia
for(i in 1:length(cortes)){
  corte_atual <- cortes[i]
  
  acuracia_atual <- mean(y_test == ifelse(prob_ridge >= corte_atual, 1, 0))
  
  df_cortes_acuracia_ridge$Corte[i] <- corte_atual
  df_cortes_acuracia_ridge$Acuracia[i] <- acuracia_atual
}

# Exibir os resultados
print(df_cortes_acuracia_ridge)

# Encontrar o corte com a maior acurácia
melhor_corte <- df_cortes_acuracia_ridge[which.max(df_cortes_acuracia_ridge$Acuracia), ]
print(paste("Melhor corte: ", melhor_corte$Corte, " com acurácia: ", melhor_corte$Acuracia, sep=""))

corte <- melhor_corte$Corte

# Plotando a área sob a curva
roc_ridge <- roc(y_test, prob_ridge)
plot(roc_ridge, main="Curva ROC", col="#1c61b6")

# Calculando a acurácia e a curva ROC com o melhor corte
resultados$acuracia[resultados$modelo == "ridge"] <- mean(y_test == ifelse(prob_ridge >= corte, 1, 0))

resultados$auc[resultados$modelo == "ridge"] <- roc(y_test, prob_ridge)$auc

resultados

vip::vip(modelo_ridge, aesthetics = list(fill = "#FF5757"))


# ------------------------------------------------------------------------------------------

# Regressão Logística com método de encolhimento Lasso ----------------------------------------------------


# Inicializar um dataframe para armazenar os resultados de corte e acurácia
df_cortes_acuracia_lasso <- data.frame(Corte=numeric(), Acuracia=numeric())

# Criar um vetor de cortes de 0.1 até 1, com incremento de 0.01
cortes <- seq(0.1, 1, by=0.01)

# Dataframe para comparar cortes e acurácia 
df_cortes_acuracia_lasso <- data.frame(Corte=numeric(length(cortes)), Acuracia=numeric(length(cortes)))

# Treinando o modelo lasso
modelo_lasso <- glmnet(x = x_tr, y = y_tr, family = "binomial", alpha = 1)

# lasso - validação cruzada -----------------------------------------------

cv_lasso <- cv.glmnet(x = x_tr, y = y_tr, alpha = 1)

plot(cv_lasso, cex.lab = 1.3)

# Treinando o modelo na base de teste
prob_lasso <- predict(modelo_lasso, newx = x_test, s = cv_lasso$lambda.1se, type = "response") # valor predito
prob_lasso <- as.numeric(prob_lasso[, 1])

# Encontrando o melhor corte
for(i in 1:length(cortes)){
  corte_atual <- cortes[i]
  
  acuracia_atual <- mean(y_test == ifelse(prob_lasso >= corte_atual, 1, 0))
  
  df_cortes_acuracia_lasso$Corte[i] <- corte_atual
  df_cortes_acuracia_lasso$Acuracia[i] <- acuracia_atual
}

# Exibir os resultados
print(df_cortes_acuracia_lasso)

# Encontrar o corte com a maior acurácia
melhor_corte <- df_cortes_acuracia_lasso[which.max(df_cortes_acuracia_lasso$Acuracia), ]
print(paste("Melhor corte: ", melhor_corte$Corte, " com acurácia: ", melhor_corte$Acuracia, sep=""))

corte <- melhor_corte$Corte

# Plotando a área sob a curva
roc_lasso <- roc(y_test, prob_lasso)
plot(roc_lasso, main="Curva ROC", col="#1c61b6")

# Calculando os valore de acurácia e área sob a curva (ROC)

resultados$acuracia[resultados$modelo == "lasso"] <- mean(y_test == ifelse(prob_lasso >= corte, 1, 0))

resultados$auc[resultados$modelo == "lasso"] <- roc(y_test, prob_lasso)$auc

resultados

vip::vip(modelo_lasso, aesthetics = list(fill = "#FF5757"))

# ------------------------------------------------------------------------------------------

# Regressão Logística com método de encolhimento elastic ----------------------------------------------------


# Inicializar um dataframe para armazenar os resultados de corte e acurácia
df_cortes_acuracia_elastic <- data.frame(Corte=numeric(), Acuracia=numeric())

# Criar um vetor de cortes de 0.1 até 1, com incremento de 0.01
cortes <- seq(0.1, 1, by=0.01)

# Dataframe para comparar cortes e acurácia 
df_cortes_acuracia_elastic <- data.frame(Corte=numeric(length(cortes)), Acuracia=numeric(length(cortes)))

# Treinando o modelo elastic_net
modelo_elastic <- glmnet(x = x_tr, y = y_tr, family = "binomial", alpha = 0.5)

# elastic - validação cruzada -----------------------------------------------

cv_elastic <- cv.glmnet(x = x_tr, y = y_tr, alpha = 0.5)

plot(cv_elastic, cex.lab = 1.3)

# Treinando o modelo elastic_net

prob_elastic <- predict(modelo_elastic, newx = x_test, s = cv_elastic$lambda.1se, type = "response") # valor predito
prob_elastic <- as.numeric(prob_elastic[, 1])

for(i in 1:length(cortes)){
  corte_atual <- cortes[i]
  
  acuracia_atual <- mean(y_test == ifelse(prob_elastic >= corte_atual, 1, 0))
  
  df_cortes_acuracia_elastic$Corte[i] <- corte_atual
  df_cortes_acuracia_elastic$Acuracia[i] <- acuracia_atual
}

# Exibir os resultados
print(df_cortes_acuracia_elastic)

# Encontrar o corte com a maior acurácia
melhor_corte <- df_cortes_acuracia_elastic[which.max(df_cortes_acuracia_elastic$Acuracia), ]
print(paste("Melhor corte: ", melhor_corte$Corte, " com acurácia: ", melhor_corte$Acuracia, sep=""))

corte <- melhor_corte$Corte

# Plotando a área sob a curva
roc_elastic <- roc(y_test, prob_elastic)
plot(roc_elastic, main="Curva ROC", col="#1c61b6")

resultados$acuracia[resultados$modelo == "elastic_net"] <- mean(y_test == ifelse(prob_elastic >= corte, 1, 0))

resultados$auc[resultados$modelo == "elastic_net"] <- roc(y_test, prob_elastic)$auc

resultados

vip::vip(modelo_elastic, aesthetics = list(fill = "#FF5757"))

# ------------------------------------------------------------------------------------------

# Árvore de regressão ----------------------------------------------------

#y_tr <- ifelse(df_tr$operating == 1, "Não", "Sim") # Muda a variável operanting 1-Não (Não faliu) 0-Sim (faliu)
#y_test <- ifelse(df_test$operating == 1, "Não", "Sim") # Muda a variável operanting 1-Não (Não faliu) 0-Sim (faliu)

# Inicializar um dataframe para armazenar os resultados de corte e acurácia
df_cortes_acuracia_arvore <- data.frame(Corte=numeric(), Acuracia=numeric())

# Criar um vetor de cortes de 0.1 até 1, com incremento de 0.01
cortes <- seq(0.1, 1, by=0.01)

# Dataframe para comparar cortes e acurácia 
df_cortes_acuracia_arvore <- data.frame(Corte=numeric(length(cortes)), Acuracia=numeric(length(cortes)))

# Modelo de árvore
library(rpart.plot)

# Validação cruzada para a árvore -------------------------------------------------------
arvore <- rpart(no_operating ~ ., data = df_tr, method = "class", control = rpart.control(xval = 10, cp = 0))
#rpart.plot(arvore, roundint = FALSE)

arvore$cptable

# Encontrando o CP ótimo para árvore
cp_ot <- arvore$cptable[which.min(arvore$cptable[,"xerror"]),"CP"]
cp_ot <- arvore$cptable %>%
  as_tibble() %>%
  filter(xerror == min(xerror))

# Rodando o modelo com o CP ótimo
poda1 <- prune(arvore, cp = cp_ot$CP[1])
rpart.plot(poda1, roundint = FALSE, main = "Árvore de Decisão")

# Treinando o modelo com o CP ótimo
prob_arvore = predict(poda1, newdata = df_test, type = "prob")
prob_arvore <- as.numeric(prob_arvore[, 2])

for(i in 1:length(cortes)){
  corte_atual <- cortes[i]
  
  acuracia_atual <- mean(y_test == ifelse(prob_arvore >= corte_atual, 1, 0))
  
  df_cortes_acuracia_arvore$Corte[i] <- corte_atual
  df_cortes_acuracia_arvore$Acuracia[i] <- acuracia_atual
}

# Exibir os resultados
print(df_cortes_acuracia_arvore)

# Encontrar o corte com a maior acurácia
melhor_corte <- df_cortes_acuracia_arvore[which.max(df_cortes_acuracia_arvore$Acuracia), ]
print(paste("Melhor corte: ", melhor_corte$Corte, " com acurácia: ", melhor_corte$Acuracia, sep=""))

corte <- melhor_corte$Corte

# Plotando a área sob a curva
roc_arvore <- roc(y_test, prob_arvore)
plot(roc_arvore, main="Curva ROC", col="#1c61b6")

resultados$acuracia[resultados$modelo == "arvore_decisao"] <- mean(y_test == ifelse(prob_arvore >= corte, 1, 0))

resultados$auc[resultados$modelo == "arvore_decisao"] <- roc(y_test, prob_arvore)$auc

resultados

vip::vip(poda1, aesthetics = list(fill = "#FF5757"))

# Plot de todas as curvas roc
plot(roc_logistica, main="Curvas ROC Comparativas", col="#1c61b6")
lines(roc_ridge, col="#b61c4a")
lines(roc_lasso, col="#2a9d8f")
lines(roc_elastic, col="#d35400")
lines(roc_arvore, col="#8e44ad")
legend("bottomright", legend=c("Logística", "Ridge", "Lasso", "Elastic Net", "Árvore"), col=c("#1c61b6", "#b61c4a", "#2a9d8f", "#d35400", "#8e44ad"), lwd=2)

#Na imagem das curvas ROC, o modelo de árvore de decisão parece ter o melhor desempenho, com a maior área sob a curva (AUC), indicando uma melhor capacidade de discriminação entre as classes positivas e negativas. O modelo logístico também apresenta uma boa performance, sendo um modelo mais simples e interpretable em comparação com a árvore de decisão. 