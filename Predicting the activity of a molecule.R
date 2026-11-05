
# Install if missing
required_packages <- c("readxl","caret","corrplot","scatterplot3d","e1071","nnet",
                       "MLmetrics","gridExtra","FSelectorRcpp","ggplot2","fastcluster",
                       "dbscan","proxy","klaR","cluster","factoextra","glmnet","Boruta",
                       "randomForest","ranger","tidyr","arules","arulesViz")
for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg)
  }
  library(pkg, character.only = TRUE)
}


# Step 0: Load and Inspect the Dataset
# Load the dataset using a file chooser dialog
molecules <- read_excel(file.choose())

# Display dataset dimensions (rows and columns)
dim(molecules)

# Display the structure of the dataset (data types, column names, etc.)
str(molecules)

# Display a statistical summary of the dataset  
summary(molecules)

# Identify non-numeric columns
non_numeric_cols <- names(molecules)[!sapply(molecules, is.numeric)]
print(non_numeric_cols)

# Change the class of the activity label to numeric
# Convert the Labels column to numeric for calculations and correlations
molecules$Labels <- factor(molecules$Labels, levels = c("0.0", "1.0", "2.0"))
molecules$Labels <- as.numeric(molecules$Labels) - 1

# Step 1: Remove the first column (index column) and duplicate entries
# Remove the first column as it does not provide meaningful information for the analysis
molecules <- molecules[, -1]
# check for duplicates
sum(duplicated(molecules))
# Keep unique entries
molecules <- unique(molecules)
nrow(molecules)

# Step 2: Check for Missing Values
# Summarize and print the number of missing values in each column
missing_values_summary <- colSums(is.na(molecules))
cat("Missing values summary:", missing_values_summary, "\n")
# comment: There are no missing values in the dataset

# Step 3: Analyze Target Variable Distribution
# Analyze the distribution of the target variable and visualize it
cat("Target variable distribution:", table(molecules$Labels), "\n")

target_sums <- as.data.frame(table(molecules$Labels))

ggplot(target_sums, aes(x = Var1, y = Freq)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  labs(title = "Distribution of Target Variable (Labels)", x = "Labels", y = "Frequency") +
  theme_minimal()

# Step 3: Calculate Variances and Means
# Calculate the variance and mean of each feature
variances <- sapply(molecules[, 1:(ncol(molecules) - 1)], var)
means <- sapply(molecules[, 1:(ncol(molecules) - 1)], mean)

# Visualize Variances and Means
# Plot the variances of features
plot(variances, main = "Variances of Features", xlab = "Feature Index", ylab = "Variance", col = "blue", pch = 16)

# Plot the means of features
plot(means, main = "Means of Features", xlab = "Feature Index", ylab = "Mean", col = "green", pch = 16)

# Step 4: Remove Columns with No Variability
# Identify and remove columns that have only one unique value
no_variability_cols <- names(molecules)[sapply(molecules, function(x) length(unique(x)) == 1)]
molecules <- molecules[, !colnames(molecules) %in% no_variability_cols]
no_variability_cols 

# Output the names of columns removed and their count
cat("Columns removed due to no variability:\n", paste(no_variability_cols, collapse = ", "), "\n")
cat("Total columns removed due to no variability:", length(no_variability_cols), "\n")




# Step 5: Remove Highly Correlated Features
# Calculate the correlation matrix for features

cor_matrix <- cor(molecules[, 1:(ncol(molecules) - 1)], method = "pearson")

# Identify highly correlated features with a cutoff of 0.9
highly_correlated <- colnames(molecules[, 1:(ncol(molecules) - 1)])[findCorrelation(cor_matrix, cutoff = 0.9)]
cat("Highly correlated features identified (correlation > 0.9):", paste(highly_correlated, collapse = ", "), "\n")

# Calculate correlations with the target variable
# Filter features strongly correlated with the target (threshold 0.2)
target_correlations <- cor(molecules[, 1:(ncol(molecules) - 1)], molecules$Labels)
strong_target_features <- colnames(molecules[, 1:(ncol(molecules) - 1)])[which(abs(target_correlations) > 0.2)]
cat("Strongly correlated features with the target (correlation > 0.2):", strong_target_features, "\n")

# Identify matching features between highly correlated and strongly correlated with the target
matching_features <- highly_correlated[highly_correlated %in% strong_target_features]
cat("Matching features between highly correlated and strong target features:\n", matching_features, "\n")

# Remove highly correlated features that are not strongly correlated with the target
features_to_remove <- highly_correlated[!highly_correlated %in% strong_target_features]
molecules <- molecules[, !(colnames(molecules) %in% features_to_remove)]

# Output the number of features removed and remaining
cat("Number of features removed due to high correlation:", length(features_to_remove), "\n")
cat("Names of features removed due to high correlation:\n", paste(features_to_remove, collapse = ", "), "\n")
cat("Number of features remaining:", ncol(molecules), "\n")

# Step 7: Visualize Correlation Matrix (Cleaned)
cor_matrix_cleaned <- cor(molecules[, 1:(ncol(molecules) - 1)], method = "pearson")

corrplot(cor_matrix_cleaned, method = "color", order = "hclust", 
         tl.col = "black", tl.cex = 0.3, type = "lower")

# Step 8: Perform PCA on Cleaned Data
# Apply Principal Component Analysis to reduce dimensions
pca_result <- prcomp(molecules[, 1:(ncol(molecules) - 1)], center = TRUE, scale. =FALSE)

# Scree plot to visualize the cumulative variance explained by principal components
pca_var <- pca_result$sdev^2 / sum(pca_result$sdev^2)
plot(cumsum(pca_var), type = "b", pch = 19, col = "blue",
     xlab = "Principal Components", ylab = "Cumulative Proportion of Variance",
     main = "Scree Plot for PCA")

# Comment: The scree plot shows that many components are required to explain 95% of the variance, 
# indicating that PCA is not effective for dimensionality reduction in this dataset.


plot(pca_result$x[,1:2], pch=20, xlab="PC1", ylab="PC2", 
     col=molecules$Labels + 1, # Adjust color based on Labels
     main="Scatter plot for the first two principal components")
legend("topright", legend=unique(molecules$Labels), 
       col=unique(molecules$Labels + 1), pch=20, title="Labels")

# Comment: Different Labels overlap significantly, so PCA does not separate the data meaningfully.


scatterplot3d(pca_result$x[,1:3], pch=20, 
              color=molecules$Labels + 1, 
              main="3D scatter plot for the first 3 principal components")
legend("topright", legend=unique(molecules$Labels), 
       col=unique(molecules$Labels + 1), pch=20, title="Labels")
# Comment: Even in 3D space, Labels do not separate clearly, showing limited usefulness of PCA.


# Direction for Next Steps
cat("Conclusion: PCA is not effective for this dataset due to overlap between Labels.
Alternative Strategy: We will explore feature selection and clustering for further analysis.
Once relevant features are selected or clusters are identified, PCA may be re-applied to refine dimensionality or for visualization purposes.")




# Step 9: Clustering Analysis

### Hierarchical Clustering ##


## we are going to try hierarchical clustering to see if there are clusters match with labels
# compute distance matrix with euclidian method since we are using pca where the results are numerical and not binary
dist_matrix <- dist(pca_result$x, method = "euclidean")
jaccard_dist <- dist(molecules[, 1: (ncol(molecules) - 1)], method = "binary")

# Clustering with Single,complete and average Linkage
hc_pca_ward.D2 <- hclust(dist_matrix, method = "ward.D2")
hc_pca_complete <- hclust(dist_matrix, method = "complete")
hc_complete <- hclust(jaccard_dist, method = "complete")
hc_ward.D2 <- hclust(jaccard_dist, method = "ward.D2")

# Visualize Complete Linkage Dendrogram
plot_pca_complete <- fviz_dend(hc_pca_complete, k=3 ,
                               main = "PCA_Complete Linkage Dendrogram", 
                               cex = 0.1, 
                               color_labels_by_k = TRUE, 
                               rect = TRUE, 
                               rect_border = "jco", 
                               rect_fill = TRUE)

plot_complete <- fviz_dend(hc_complete, k=3 ,
                           main = "Complete Linkage Dendrogram", 
                           cex = 0.1, 
                           color_labels_by_k = TRUE, 
                           rect = TRUE, 
                           rect_border = "jco", 
                           rect_fill = TRUE)

# Visualize ward.D2 Linkage Dendrogram
plot_pca_ward.D2 <- fviz_dend(hc_pca_ward.D2,  k=3,
                              main = "PCA_ward.D2 Dendrogram", 
                              cex = 0.1, 
                              color_labels_by_k = TRUE, 
                              rect = TRUE, 
                              rect_border = "jco", 
                              rect_fill = TRUE) 

plot_ward.D2 <- fviz_dend(hc_ward.D2,  
                          main = "ward.D2 Dendrogram", k=3,
                          cex = 0.1, 
                          color_labels_by_k = TRUE, 
                          rect = TRUE, 
                          rect_border = "jco", 
                          rect_fill = TRUE) 


plot_pca_complete
plot_complete
plot_pca_ward.D2
plot_ward.D2



hc_ward.D2_clusters <- cutree(hc_ward.D2,k=3)
hc_pca_ward.D2_clusters <- cutree(hc_pca_ward.D2,k=3)
hc_complete_clusters <- cutree(hc_complete, k = 3)
hc_pca_complete_clusters <- cutree(hc_pca_complete, k = 3)


confusion_matrix1 <- table(hc_pca_ward.D2_clusters, molecules$Labels)
confusion_matrix2 <- table(hc_ward.D2_clusters, molecules$Labels)
confusion_matrix3 <- table(hc_complete_clusters, molecules$Labels)
confusion_matrix4 <- table(hc_pca_complete_clusters, molecules$Labels)


# Function to convert confusion matrix to a ggplot object
plot_confusion_matrix <- function(cm, title) {
  cm_df <- as.data.frame(cm)
  colnames(cm_df) <- c("Predicted", "Actual", "Freq")
  
  ggplot(cm_df, aes(x = Actual, y = Predicted, fill = Freq)) +
    geom_tile(color = "white") +
    geom_text(aes(label = Freq), color = "black", size = 3.5) +
    scale_fill_gradient(low = "white", high = "steelblue") +
    labs(title = title, x = "Actual", y = "Predicted") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}

# Generate individual plots
p1 <- plot_confusion_matrix(confusion_matrix1, "PCA Ward.D2")
p2 <- plot_confusion_matrix(confusion_matrix2, "Ward.D2")
p3 <- plot_confusion_matrix(confusion_matrix3, "Complete")
p4 <- plot_confusion_matrix(confusion_matrix4, "PCA Complete")

# visualize together all the confusion matrices
grid.arrange(p1, p2, p3, p4, ncol = 2)



sil_hc_pca_ward.D2 <- silhouette(hc_pca_ward.D2_clusters, dist_matrix)
sil_hc_ward.D2 <- silhouette(hc_ward.D2_clusters, jaccard_dist)
sil_hc_pca_complete <- silhouette(hc_pca_complete_clusters, dist_matrix)
sil_hc_complete <- silhouette(hc_complete_clusters, jaccard_dist)

fviz_silhouette(sil_hc_pca_complete) + 
  ggtitle("Silhouette Analysis for Hierarchical Clustering complete method after PCA")

fviz_silhouette(sil_hc_complete) + 
  ggtitle("Silhouette Analysis for Hierarchical Clustering complete method")

fviz_silhouette(sil_hc_pca_ward.D2) + 
  ggtitle("Silhouette Analysis for Hierarchical Clustering ward.D2 method after PCA")

fviz_silhouette(sil_hc_ward.D2) + 
  ggtitle("Silhouette Analysis for Hierarchical Clustering ward.D2 method")


cat("
Conclusions about the hierarchical clustering analysis:

1. The confusion matrices show that no cluster clearly matches a single true label (0, 1, or 2). 
   In other words, each of the 3 clusters contains a mixture of actual class labels.
   So the clusters we get from hierarchical clustering do not map onto the known activity labels in the dataset.
   
2. Ward.D2 with Jaccard might gather one or two groups that mostly share one label, but also “steals” points from other labels, lowering purity.
   Ward.D2 with PCA tries to cluster on the major principal component directions,
   sometimes leading to slightly different label distributions in each cluster, but also exhibiting overlap.
   Complete Linkage has a similar pattern: each cluster is a mix of 0, 1, 2.

3. All methods yield silhouette widths mostly in the 0.0–0.25 range, suggesting weak separation.
   Values that approach or exceed 0.5 typically indicate well-defined clusters.
   Here, the dataset does not separate neatly into 3 distinct clusters with either the original features (using Jaccard)
   or the PCA-transformed features (using Euclidean)

4. For Ward.D2 on Jaccard, one cluster did achieve ~0.25, which is the best of the scores.
   This indicates that for binary fingerprint data, Jaccard is often more appropriate than Euclidean on PCA scores.
   
5. Whether we use Ward.D2 or Complete Linkage, with PCA (Euclidean) or without PCA (Jaccard),
   the hierarchical clustering solution has weak consistency with the known labels and low average silhouette values.")   




### K-Means Clustering ###

set.seed(123)

kmeans_result <- kmeans(molecules[, 1: (ncol(molecules) - 1)], centers = 3, nstart = 25)
kmeans_pca_result <- kmeans(pca_result$x[,1:60], centers = 3, nstart = 25)


# visualize K-Means

# K-Means plot represents clustering in the original feature space.
kmeans_plot <-fviz_cluster(kmeans_result, data = molecules[, 1: (ncol(molecules) - 1)], 
                           geom = "point",
                           ellipse.type = "convex", 
                           main = "K-Means Clustering",
                           palette = c("red", "blue", "green","black"), 
                           ggtheme = theme_minimal()
)
kmeans_plot

kmeans_plot + 
  geom_text(aes(label = molecules$Labels, color = factor(molecules$Labels)),
            vjust = -1,
            size = 2,
            show.legend = FALSE)

# K-Means plot shows clustering after PCA transformation.
kmeans_pca_plot <-fviz_cluster(kmeans_pca_result, data = pca_result$x[,1:60], 
                               geom = "point",
                               ellipse.type = "convex", # Convex hull around clusters
                               main = "K-Means Clustering after PCA",
                               palette = c("red", "blue", "green","black"), 
                               ggtheme = theme_minimal()
)
kmeans_pca_plot
kmeans_pca_plot + 
  geom_text(aes(label = molecules$Labels, color = factor(molecules$Labels)),
            vjust = -1,
            size = 2,
            show.legend = FALSE)

# overlay the true labels on the PCA plot to assess alignment with actual molecular activity.                  
fviz_cluster(list(data = pca_result$x[,1:60], cluster = molecules$Labels),
             geom = "point",         
             ellipse.type = "convex",
             main = "Clusters by True Labels", 
             palette = c("red", "blue", "green"),        
             ggtheme = theme_minimal()
)

cat(" comments:
- K-Means Before PCA:
    #The clusters appear to overlap significantly.
    #Some degree of separation is observed, but the boundaries are not well-defined.

- K-Means After PCA:
    #PCA reduces dimensions while preserving variance, allowing for better separation.
    #More distinct cluster formations are visible, with reduced overlap.

- Comparison with True Labels:
    #Even though PCA improves clustering separation, clusters do not perfectly match the true labels.
    #The actual class labels are mixed across different clusters, indicating that K-Means alone may not be the best clustering approach for this dataset.
    
- I have to mention that we got those results using only the first two principal components who explain no more than 35% of the data.
  when we used full PCA spacethe results and the ovelaping of the clusters was even worse. In both cases, K-Means does not produce clusters
  that match the 0, 1, 2 labels, indicating only partial separation at best.  ")



# Compute Silhouette Scores for K-Means on Original Data
sil_kmeans <- silhouette(kmeans_result$cluster, jaccard_dist)
fviz_silhouette(sil_kmeans) + 
  ggtitle("Silhouette Analysis for K-Means Clustering (Original data)")

# Compute Average Silhouette Score for Original K-Means
avg_sil_kmeans <- mean(sil_kmeans[, 3])
cat("K-Means Silhouette Average:", round(avg_sil_kmeans, 3), "\n\n")

# Compute Silhouette Scores for K-Means on PCA Data
sil_kmeans_pca <- silhouette(kmeans_pca_result$cluster, dist_matrix)
fviz_silhouette(sil_kmeans_pca) + 
  ggtitle("Silhouette Analysis for K-Means Clustering (pca)")

# Compute Average Silhouette Score for PCA K-Means
avg_sil_kmeans_pca <- mean(sil_kmeans_pca[, 3])
cat("K-Means Silhouette Average (PCA):", round(avg_sil_kmeans_pca, 3), "\n\n")


cat("comments:
- These silhouette results of K-Means does not strongly isolate the three known label groups.
  Even after dimensionality reduction with PCA, clusters remain overlapping, with silhouette scores under 0.20.
  This suggests the data may not have a natural 3-cluster structure aligned with the labels
    
- Clustering without PCA (using Jaccard distance) has a slightly better overall silhouette (~0.14) than clustering with PCA (~0.10).
  So for binary molecular fingerprints, the Jaccard distance can sometimes capture the data’s structure
  better than Euclidean distance in the PCA-transformed space. ")




### ANOVA ###
# Change Labels to be factor
molecules$Labels <- factor(molecules$Labels)
# anova for every feature
feature_cols <- colnames(molecules)[1:(ncol(molecules) - 1)]

# Create a data frame to store results
results_anova <- data.frame(Feature = character(), PValue = numeric())

for (feat in feature_cols) {
  # One-way ANOVA for "feature ~ label"
  model   <- aov(molecules[[feat]] ~ molecules$Labels)
  p_value <- summary(model)[[1]][["Pr(>F)"]][1]
  
  results_anova <- rbind(results_anova, 
                         data.frame(Feature = feat, PValue = p_value))
}

# sort by ascending p-value
results_anova <- results_anova[order(results_anova$PValue), ]
head(results_anova, 20)  

# 1. Identify the top 10 features with the smallest p-values
top15_feats <- results_anova$Feature[1:15]

# 2. For each of these features, create a boxplot
par(mfrow = c(3, 5))  # optional: arrange plots in a 2-row by 5-column layout

for (feat in top15_feats) {
  boxplot(molecules[[feat]] ~ molecules$Labels,
          main = feat,
          xlab = "Labels",
          ylab = "Feature Value",
          col  = c("lightblue","lightgreen","lightpink"))
}

cat("comments : 
Since the dataset contains three groups (0, 1, 2), a t-test would not be appropriate for confirming group differences
One-way ANOVA was performed for each feature to check whether there are statistically significant differences between labels.
Features with small p-values (< 0.05) indicate significant variation among groups, meaning they can help classify molecular activity.
Top discriminative features found using ANOVA include NS, CSN, QS, A!N$A, C%N, which had extremely small p-values (e.g., p < 1e-30).
These features exhibit clear separation between at least one pair of labels, making them crucial for classification.
However, examinating the boxplots some features showed overlapping distributions between classes, making them less useful for classification.")

table(molecules$`QAAA@1`,molecules$Labels)


### Multinomial Logistic Regression on Full Dataset ###

set.seed(123)
##Split into Training and Test Sets with all the features of the dataset
train_index <- createDataPartition(molecules$Labels, p = 0.7, list = FALSE)
train_data  <- molecules[train_index, ]
test_data   <- molecules[-train_index, ]


# Fit a multinomial model using all features
model_all <- multinom(Labels ~ ., data = train_data, maxit=200)

# Predict on the test set
pred_all <- predict(model_all, newdata = test_data)

conf_matrix_all <- confusionMatrix(pred_all, test_data$Labels)
conf_matrix_all  


###  Multinomial Logistic Regression on ANOVA-selected Features ###

## Split into Training and Test Sets with best features indicated in ANOVA

train_top15_feats  <- train_data[, c(top15_feats, "Labels")]
test_top15_feats   <- test_data[, c(top15_feats, "Labels")]


# Fit a multinomial model using all features
model_top15_feats <- multinom(Labels ~ ., data = train_top15_feats, maxit=200)

# Predict on the test set
pred_top15_feats <- predict(model_top15_feats, newdata = test_top15_feats)

conf_matrix_top15_feats <- confusionMatrix(pred_top15_feats, test_top15_feats$Labels)
conf_matrix_top15_feats


cat("Comments:
We performed Multinomial Logistic Regression twice: once using the full feature set of the dataset, and once using only the top 15 features selected via one-way ANOVA.
The results show that the model using all features achieved better performance, with 64.3% accuracy and a Kappa score of 0.35, compared to 60.8% accuracy and a Kappa of 0.21 using the ANOVA-selected features.
This suggests that by reducing the feature set to only 15 variables, important information may have been lost, leading to reduced classification performance.
Therefore, to better identify the optimal feature subset, we proceed with applying four different feature selection methods (Lasso, Boruta, RFE, and Mutual Information). 
By evaluating models built with different numbers of top mutual features, we aim to determine the optimal balance between dimensionality reduction and model effectiveness.")




### Section 1: Feature Selection - LASSO  ###

x <- as.matrix(molecules[, 1:(ncol(molecules) - 1)])
y <- molecules$Labels

set.seed(234)
# Fit LASSO with standardization OFF (critical for binary features)
lasso_model <- cv.glmnet(x , y,  alpha = 1,
                         family = "multinomial",  # Multinomial for 3+ classes
                         type.measure = "class",
                         standardize = FALSE  )    # Disable internal standardization

# Extract coefficients at lambda.min (minimizes cross-validation error)
lasso_coef <- coef(lasso_model, s = "lambda.min")

# Combine coefficients across classes
coef_matrix <- do.call(cbind, lapply(lasso_coef, as.matrix))

# Aggregate feature importance: MAX absolute value across classes
coef_agg <- apply(abs(coef_matrix), 1, max)  # Emphasizes class-specific importance

# Remove intercept (first row) and get feature names
features <- rownames(coef_matrix)[-1]
coef_agg <- coef_agg[-1]
names(coef_agg) <- features

# Get top 20 features
top_20_lasso <- names(sort(coef_agg, decreasing = TRUE))[1:20]
print(top_20_lasso)
top_20_scores<- coef_agg[top_20_lasso]
top_20_scores

# Create a data frame for plotting
plot_data <- data.frame(
  Feature = factor(top_20_lasso, levels = top_20_lasso),  # Preserve order
  Importance = top_20_scores)

# Plot the top 20 features
ggplot(plot_data, aes(x = Feature, y = Importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  labs(
    title = "Top 20  Features by Lasso Importance",
    x = "Feature",
    y = "Importance Score"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1),  # Rotate x-axis labels
        text = element_text(size = 8),  
        axis.text = element_text(size =10),  
        axis.title = element_text(size =10),  
        plot.title = element_text(size = 15)) 



### Section 2: Feature Selection - Boruta ###

set.seed(345)  
boruta_model <- Boruta(Labels~. , data= molecules , maxRuns = 200,    doTrace = 2   )

# Get confirmed important features
confirmed_features <- getSelectedAttributes(boruta_model, withTentative = FALSE)
importance <- attStats(boruta_model)

# Extract top 10 features
sorted_features <- rownames(importance)[order(-importance$meanImp)]
top_20_boruta <- sorted_features[1:20]
print(top_20_boruta)


# Plot feature importance
feature_importance <- data.frame(
  Feature = rownames(importance),
  Importance = importance$meanImp)

# Filter the data frame to include only the top 20 features
top20_boruta <- feature_importance[feature_importance$Feature %in% top_20_boruta, ]

ggplot(top20_boruta, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Top Features by Boruta Importance", 
       x = "Feature", 
       y = "Mean Importance (Z-score)") +
  theme_minimal()+
  theme(text = element_text(size = 8),  
        axis.text = element_text(size =10),  
        axis.title = element_text(size =10),  
        plot.title = element_text(size = 15))



###  Section 3: Feature Selection - RFE ###


# RFE control
set.seed(123)  
ctrl <- rfeControl(  functions = rfFuncs,    method = "cv",   number = 10  )


# Perform RFE
rfe_result <- rfe(  x,  y,  sizes = c(10, 20, 30),  rfeControl = ctrl)


# Top 20 features
top_20_rfe <- predictors(rfe_result)[1:20]


print(top_20_rfe )

# Extract feature importance scores
feature_importance <- varImp(rfe_result)

# Convert to data frame and filter the top 10 features
feature_importance_df <- data.frame(
  Feature = rownames(feature_importance),
  Importance = feature_importance$Overall)

# Sort by importance and keep only the top 20
top_20_df <- feature_importance_df[order(-feature_importance_df$Importance), ][1:20, ]

# Plot the top 20 features

ggplot(top_20_df, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Top 10 Features by RFE Importance", 
       x = "Feature", 
       y = "Importance Score") +
  theme_minimal() +
  theme(text = element_text(size = 8),  
        axis.text = element_text(size = 10),  
        axis.title = element_text(size = 10),  
        plot.title = element_text(size = 15))




###  Section 4: Feature Selection - Mutual Information ###

# Compute Mutual Information for each feature

set.seed(567)
mi_scores <- information_gain(x=as.data.frame(x),y=y, type = "symuncert")

# Select the top 20 features 
mi_scores <- mi_scores[order(-mi_scores$importance), ]

# Print selected features
print(head(mi_scores, 20))
top_20_mi <- head(mi_scores$attributes,20)
top_20_mi


#visualize results
ggplot(head(mi_scores, 20), aes(x = reorder(attributes, importance), y = importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Top Features by Mutual Information ", 
       x = "Feature", 
       y = "Mutual Information Score") +
  theme_minimal() + 
  theme(text = element_text(size = 8),  
        axis.text = element_text(size = 10),  
        axis.title = element_text(size =10),  
        plot.title = element_text(size = 15))






### Section 5: Combine Top Mutual Features Across Methods ###


top_120_lasso <- names(sort(coef_agg, decreasing = TRUE))[1:120]
top_120_boruta <- sorted_features[1:120]
top_120_rfe <- predictors(rfe_result)[1:120]
top_120_mi <- head(mi_scores$attributes,120)

# We are going to create a data frame including top 120 important features from each model we runed

model_results <- data.frame(LASSO=top_120_lasso,Boruta=top_120_boruta,RFE=top_120_rfe,Mutual_info=top_120_mi)

compined <- c(top_120_lasso,top_120_boruta,top_120_rfe,top_120_mi)
count_mutual_features <- table(compined)
sorted_mutual_features <- sort(count_mutual_features,decreasing = TRUE)
head(sorted_mutual_features,120)

important_features <- rownames(head(sorted_mutual_features,120))
top_features_selected <- molecules[,c(important_features,"Labels")]

cat("Comments:
We took the top 120 features from each method (Lasso, Boruta, RFE, and Mutual Information) and put them all together in one list.
Then we counted how many times each feature appeared across the methods. The features that showed up more often are likely more important.
We'll use these 'mutual' features in the next steps to test different feature counts and find out how many features give us the best results.")


### Section 8: Multinomial Logistic Regression After Feature Selection ###

set.seed(123)
## Split into Training and Test Sets with  the features of the top_features_selected
train_index <- createDataPartition(top_features_selected$Labels, p = 0.7, list = FALSE)
train_data_top_features  <- top_features_selected[train_index,]
test_data_top_features  <- top_features_selected[-train_index,]


# Fit a multinomial model using top_features_selected
model_top_features_selected <- multinom(Labels ~ ., data = train_data_top_features , maxit=200,class.weights = c("0" = 0.7, "1" = 1.1, "2" = 1.8))

# Predict on the test set
pred_top_features_selected<- predict(model_top_features_selected, newdata = test_data_top_features)

conf_matrix_top_features_selected <- confusionMatrix(pred_top_features_selected, test_data_top_features$Labels)
conf_matrix_top_features_selected 

cat("Comments:
Here we trained a multinomial logistic regression model using the 120 mutual top features from all selection methods, and we also applied class weights to handle label imbalance.
Compared to our earlier models (one with all features and one with only 15 ANOVA-selected features), this one gives better results — around 65% accuracy and a Kappa of 0.35.
So, selecting important features and adjusting for imbalance really helped improve performance.")




### Section 9: SVM Model Comparisons (with and without weighting) ###


svm_model <- svm(Labels ~ ., data = train_data, kernel = "radial",class.weights = c("0" = 0.7, "1" = 1.1, "2" = 1.8))
svm_pred <- predict(svm_model, test_data)
confusionMatrix(svm_pred, test_data$Labels)

svm_model <- svm(Labels ~ ., data = train_data, kernel = "radial")
svm_pred <- predict(svm_model, test_data)
confusionMatrix(svm_pred, test_data$Labels)

svm_model <- svm(Labels ~ ., data = train_data_top_features, kernel = "radial",class.weights = c("0" = 0.7, "1" = 1.1, "2" = 1.8))
svm_pred <- predict(svm_model, test_data_top_features)
confusionMatrix(svm_pred, test_data_top_features$Labels)

svm_model <- svm(Labels ~ ., data = train_data_top_features, kernel = "radial")
svm_pred <- predict(svm_model, test_data_top_features)
confusionMatrix(svm_pred, test_data_top_features$Labels)


cat("Comments:
We trained an SVM model in four different ways to compare results:
1. Using all features with class weights,
2. Using all features without weights,
3. Using top selected features with weights,
4. Using top features without weights.

What we found:
- All models had similar overall accuracy (around 70%).
- The models without weights did a bit better overall in accuracy.
- The models with weights were better at detecting the smaller classes (like class 1 and 2), but they made a few more mistakes on class 0.
- Using just the top features worked almost as well as using all features, which is great — it means we can simplify the model without losing much performance.

Next, we can try tuning the SVM settings or testing other models to improve even more.")




### Section 10: SVM Evaluation Loop on Fixed Mutual Ranking


evaluate_svm_on_mutual_features <- function(data, mutual_feature_ranking, feature_counts = c(20,30,40,50,60,70,80, 90,100,110,120), 
                                            class_weights = NULL, seed = 123) {
  
  results <- data.frame()
  
  for (n in feature_counts) {
    
    cat("🔁 Evaluating SVM with top", n, "mutual features...\n")
    
    #  Select top N features + label
    selected_features <- c(mutual_feature_ranking[1:n], "Labels")
    data_selected <- data[, selected_features]
    
    #  Train/test split
    set.seed(seed)
    train_index <- createDataPartition(data_selected$Labels, p = 0.7, list = FALSE)
    train_data <- data_selected[train_index, ]
    test_data  <- data_selected[-train_index, ]
    
    # Train SVM with optional class weights
    svm_model <- svm(Labels ~ .,data = train_data,kernel = "radial",class.weights = class_weights )
    
    #  Predict and evaluate
    predictions <- predict(svm_model, test_data)
    cm <- confusionMatrix(predictions, test_data$Labels)
    
    # Collect performance metrics
    acc     <- cm$overall["Accuracy"]
    kappa   <- cm$overall["Kappa"]
    bal_acc <- mean(cm$byClass[, "Balanced Accuracy"])
    sens0 <- cm$byClass["Class: 0", "Sensitivity"]
    sens1   <- cm$byClass["Class: 1", "Sensitivity"]
    sens2   <- cm$byClass["Class: 2", "Sensitivity"]
    
    #  Macro F1-Score
    classes <- levels(test_data$Labels)
    F1_scores <- sapply(classes, function(cl) {
      F1_Score(y_pred = as.character(predictions), 
               y_true = as.character(test_data$Labels), positive = cl)
    })
    macro_F1 <- mean(F1_scores)
    
    #  Save results
    results <- rbind(results, data.frame(
      Top_Features = n,
      Accuracy = round(acc, 4),
      Kappa = round(kappa, 4),
      Balanced_Accuracy = round(bal_acc, 4),
      Sensitivity_Class0 = round(sens0, 4),
      Sensitivity_Class1 = round(sens1, 4),
      Sensitivity_Class2 = round(sens2, 4),
      Macro_F1 = round(macro_F1, 4)
    ))
  }
  
  return(results)
}

# How many features to test
feature_counts <- c(20,30,40,50,60,70,80, 90,100,110,120)

# Class weights based on imbalance
class_weights <- c("0" = 0.7, "1" = 1.1, "2" = 1.8)

# Run evaluation loop
mutual_svm_results <- evaluate_svm_on_mutual_features(
  data = molecules,
  mutual_feature_ranking = important_features,
  feature_counts = feature_counts,
  class_weights = class_weights
)

# Show results
print(mutual_svm_results)





###  Section 11: SVM Evaluation on Dynamic Mutual Features ###



evaluate_svm_dynamic_mutual_features <- function(data, 
                                                 lasso_features, boruta_features, rfe_features, mi_features,
                                                 feature_counts = c(20,30,40,50,60,70,80, 90,100,110,120), class_weights = NULL, seed = 123) {
  
  results <- data.frame()
  
  for (n in feature_counts) {
    cat("🔎 Evaluating with top", n, "dynamic mutual features...\n")
    
    # Get the top N features from each selection method
    top_N_lasso  <- lasso_features[1:n]
    top_N_boruta <- boruta_features[1:n]
    top_N_rfe    <- rfe_features[1:n]
    top_N_mi     <- mi_features[1:n]
    
    # Combine and count frequencies
    combined_N <- c(top_N_lasso, top_N_boruta, top_N_rfe, top_N_mi)
    freq_table <- table(combined_N)
    sorted_freq <- sort(freq_table, decreasing = TRUE)
    
    # Select the top n mutual features (from the frequency count)
    selected_features <- names(head(sorted_freq, n))
    
    # Subset data to these features plus Labels
    data_selected <- data[, c(selected_features, "Labels")]
    
    # Split into training and test sets
    set.seed(seed)
    train_idx <- createDataPartition(data_selected$Labels, p = 0.7, list = FALSE)
    train_data <- data_selected[train_idx, ]
    test_data  <- data_selected[-train_idx, ]
    
    # Train SVM with the specified class weights
    svm_model <- svm(Labels ~ ., data = train_data, kernel = "radial",
                     class.weights = class_weights)
    
    # Make predictions
    predictions <- predict(svm_model, test_data)
    
    # Confusion Matrix evaluation
    cm <- confusionMatrix(predictions, test_data$Labels)
    acc     <- cm$overall["Accuracy"]
    kappa   <- cm$overall["Kappa"]
    # For balanced accuracy, we average the class-specific balanced accuracies:
    bal_acc <- mean(cm$byClass[, "Balanced Accuracy"])
    sens0 <- cm$byClass["Class: 0", "Sensitivity"]
    sens1   <- cm$byClass["Class: 1", "Sensitivity"]
    sens2   <- cm$byClass["Class: 2", "Sensitivity"]
    
    # Calculate Macro F1-Score: average of F1 scores for each class
    classes <- levels(test_data$Labels)
    F1_scores <- sapply(classes, function(cl) {
      F1_Score(y_pred = as.character(predictions), 
               y_true = as.character(test_data$Labels), positive = cl)
    })
    macro_F1 <- mean(F1_scores)
    
    # Store results
    results <- rbind(results, data.frame(
      Top_Features = n,
      Accuracy = round(acc, 4),
      Kappa = round(kappa, 4),
      Balanced_Accuracy = round(bal_acc, 4),
      Sensitivity_Class0 = round(sens0, 4),
      Sensitivity_Class1 = round(sens1, 4),
      Sensitivity_Class2 = round(sens2, 4),
      Macro_F1 = round(macro_F1, 4)
    ))
  }
  
  return(results)
}


# Define the numbers of top features to test:
feature_counts <- c(20,30,40,50,60,70,80, 90,100,110,120)

# Define class weights (example values based on your imbalance):
class_weights <- c("0" = 0.7, "1" = 1.1, "2" = 1.8)

# Run the evaluation (using your precomputed top 110 lists)
dynamic_results <- evaluate_svm_dynamic_mutual_features(
  data = molecules,
  lasso_features = top_120_lasso,
  boruta_features = top_120_boruta,
  rfe_features = top_120_rfe,
  mi_features = top_120_mi,
  feature_counts = feature_counts,
  class_weights = class_weights,
  seed = 123
)

print(dynamic_results)



cat("comments:
we created two approaches to test how many features give the best results in SVM:
1. Fixed Mutual Ranking:
   We merged the top 120 features from all methods, counted how often each feature appeared, 
   and kept the top-ν (e.g., top 20, 30, 40…) from this single merged list for every model.

2. Dynamic Mutual Ranking:
   For each n (20, 30, 40…), we got the top n features from each method and re-checked 
   which features appear in multiple methods. Then we trained a model with those dynamic mutual top-n features.

For each approach, we trained SVM models with class weighting (to handle imbalance), and recorded Accuracy, Kappa, Balanced Accuracy, and Macro F1-score to compare them.
When we use fixed mutual features (from the big merged top-120 list), the model improves steadily up to 90–100 features, but performance flattens after that.

The dynamic approach (re-evaluating the top-n most mutual features for each step) consistently performs better.
Best Macro F1 and Accuracy were achieved with the dynamic approach at 70-80 features:
Accuracy: 0.7089, Macro F1: 0.6513

Fixed method had its best at 90–100 features, but still underperformed slightly.

So, choosing the best mutual features for each n-size subset gives better performance overall.
After ploting the results to visualize this comparison we will keep this strategy to check more models ")


# Add method column to both results
mutual_svm_results$Method <- "Fixed"
dynamic_results$Method <- "Dynamic"

# Combine into one long-format dataframe
comparison_df <- rbind(mutual_svm_results, dynamic_results)


ggplot(comparison_df, aes(x = Top_Features, y = Accuracy, color = Method)) +
  geom_line(size = 1.2) +
  geom_point(size = 2) +
  labs(title = "SVM Accuracy vs. Number of Features",
       x = "Number of Top Mutual Features",
       y = "Accuracy",
       color = "Selection Method") +
  theme_minimal() +
  theme(text = element_text(size = 12))

ggplot(comparison_df, aes(x = Top_Features, y = Macro_F1, color = Method)) +
  geom_line(size = 1.2) +
  geom_point(size = 2) +
  labs(title = "SVM Macro F1 Score vs. Number of Features",
       x = "Number of Top Mutual Features",
       y = "Macro F1-Score",
       color = "Selection Method") +
  theme_minimal() +
  theme(text = element_text(size = 12))





###  Section 12: MLR Evaluation on Dynamic Mutual Features  ###



evaluate_mlr_dynamic_mutual_features <- function(data, 
                                                 lasso_features, boruta_features, rfe_features, mi_features,
                                                 feature_counts = c(20,30,40,50,60,70,80, 90,100,110,120), seed = 123) {
  
  results <- data.frame()
  
  for (n in feature_counts) {
    cat("🔎 Evaluating MLR with top", n, "mutual features...\n")
    
    # Top N from each method
    top_N_lasso  <- lasso_features[1:n]
    top_N_boruta <- boruta_features[1:n]
    top_N_rfe    <- rfe_features[1:n]
    top_N_mi     <- mi_features[1:n]
    
    # Combine & count frequencies
    combined_N <- c(top_N_lasso, top_N_boruta, top_N_rfe, top_N_mi)
    freq_table <- table(combined_N)
    sorted_freq <- sort(freq_table, decreasing = TRUE)
    
    # Select top mutual features
    selected_features <- names(head(sorted_freq, n))
    data_selected <- data[, c(selected_features, "Labels")]
    
    # Train/Test split
    set.seed(seed)
    idx <- createDataPartition(data_selected$Labels, p = 0.7, list = FALSE)
    train <- data_selected[idx, ]
    test  <- data_selected[-idx, ]
    
    # Train multinomial model
    model <- multinom(Labels ~ ., data = train, maxit = 200, trace = FALSE)
    
    # Predict and evaluate
    preds <- predict(model, test)
    cm <- confusionMatrix(preds, test$Labels)
    
    acc     <- cm$overall["Accuracy"]
    kappa   <- cm$overall["Kappa"]
    bal_acc <- mean(cm$byClass[, "Balanced Accuracy"])
    sens0 <- cm$byClass["Class: 0", "Sensitivity"]
    sens1   <- cm$byClass["Class: 1", "Sensitivity"]
    sens2   <- cm$byClass["Class: 2", "Sensitivity"]
    
    # Macro F1
    classes <- levels(test$Labels)
    F1_scores <- sapply(classes, function(cl) {
      F1_Score(y_pred = as.character(preds), y_true = as.character(test$Labels), positive = cl)
    })
    macro_F1 <- mean(F1_scores)
    
    results <- rbind(results, data.frame(
      Top_Features = n,
      Accuracy = round(acc, 4),
      Kappa = round(kappa, 4),
      Balanced_Accuracy = round(bal_acc, 4),
      Sensitivity_Class0 = round(sens0, 4),
      Sensitivity_Class1 = round(sens1, 4),
      Sensitivity_Class2 = round(sens2, 4),
      Macro_F1 = round(macro_F1, 4)
    ))
  }
  
  return(results)
}



mlr_results <- evaluate_mlr_dynamic_mutual_features(
  data = molecules,
  lasso_features = top_120_lasso,
  boruta_features = top_120_boruta,
  rfe_features = top_120_rfe,
  mi_features = top_120_mi,
  feature_counts = c(20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120)
)

print(mlr_results)






###  Section 13:  Forest Ranger Evaluation on Dynamic Mutual Features  ###



evaluate_rf_dynamic_mutual_features <- function(data, 
                                                lasso_features, boruta_features, rfe_features, mi_features,
                                                feature_counts = c(20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120),
                                                class_weights = NULL,
                                                seed = 123) {
  
  results <- data.frame()
  
  for (n in feature_counts) {
    cat(" Evaluating RF (ranger) with top", n, "mutual features...\n")
    
    # Top N from each method
    top_N_lasso  <- lasso_features[1:n]
    top_N_boruta <- boruta_features[1:n]
    top_N_rfe    <- rfe_features[1:n]
    top_N_mi     <- mi_features[1:n]
    
    combined_N <- c(top_N_lasso, top_N_boruta, top_N_rfe, top_N_mi)
    freq_table <- table(combined_N)
    sorted_freq <- sort(freq_table, decreasing = TRUE)
    
    selected_features <- names(head(sorted_freq, n))
    data_selected <- data[, c(selected_features, "Labels")]
    colnames(data_selected) <- make.unique(make.names(colnames(data_selected)))
    
    # Split train/test
    set.seed(seed)
    idx <- createDataPartition(data_selected$Labels, p = 0.7, list = FALSE)
    train <- data_selected[idx, ]
    test  <- data_selected[-idx, ]
    
    # Train using ranger
    ranger_model <- ranger(Labels ~ ., data = train, probability = FALSE,
                           num.trees = 500,
                           class.weights = class_weights,
                           seed = seed)
    
    # Predict
    preds <- predict(ranger_model, data = test)$predictions
    
    # Evaluation
    cm <- confusionMatrix(preds, test$Labels)
    acc     <- cm$overall["Accuracy"]
    kappa   <- cm$overall["Kappa"]
    bal_acc <- mean(cm$byClass[, "Balanced Accuracy"])
    sens0 <- cm$byClass["Class: 0", "Sensitivity"]
    sens1 <- cm$byClass["Class: 1", "Sensitivity"]
    sens2 <- cm$byClass["Class: 2", "Sensitivity"]
    
    classes <- levels(test$Labels)
    F1_scores <- sapply(classes, function(cl) {
      F1_Score(y_pred = as.character(preds), y_true = as.character(test$Labels), positive = cl)
    })
    macro_F1 <- mean(F1_scores)
    
    results <- rbind(results, data.frame(
      Top_Features = n,
      Accuracy = round(acc, 4),
      Kappa = round(kappa, 4),
      Balanced_Accuracy = round(bal_acc, 4),
      Sensitivity_Class0 = round(sens0, 4),
      Sensitivity_Class1 = round(sens1, 4),
      Sensitivity_Class2 = round(sens2, 4),
      Macro_F1 = round(macro_F1, 4)
    ))
  }
  
  return(results)
}


# Class weights 
class_weights <- c("0" = 0.7, "1" = 1.1, "2" = 1.8)

rf_results_ranger <- evaluate_rf_dynamic_mutual_features(
  data = molecules,
  lasso_features = top_120_lasso,
  boruta_features = top_120_boruta,
  rfe_features = top_120_rfe,
  mi_features = top_120_mi,
  feature_counts = c(20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120),
  class_weights = class_weights
)

print(rf_results_ranger)


cat("comments:
We tested three classification models : SVM, Multinomial Logistic Regression (MLR), and Random Forest using ranger,
using dynamically selected mutual features, trying different numbers of features (from 20 to 120).

Both SVM and Random Forest gave strong results, especially as we increased the number of features.
Random Forest, though, generally performed a bit better than SVM, especially in terms of both Accuracy and Macro F1 score.
The best performance overall came from Random Forest with 90 features, reaching 72.6% accuracy and a 0.66 Macro F1,
which makes it the top choice for this task.

MLR didn’t perform as well compared to the other two models. It’s worth mentioning that MLR doesn’t support class weights in this case,
so it might have been at a disadvantage — something to keep in mind when comparing results. ")



## Plots for comparison

# Create a dataframe with  results
results_df <- data.frame(
  Top_Features = c(20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120),
  SVM_Accuracy = c(0.6841, 0.7001, 0.7074, 0.7074, 0.6987, 0.7089, 0.7089, 0.7074, 0.7103, 0.7074, 0.6943),
  SVM_MacroF1 = c(0.6036, 0.6293, 0.6392, 0.6377, 0.6363, 0.6503, 0.6513, 0.6496, 0.6472, 0.6377, 0.6226),
  SVM_Kappa = c(0.4242, 0.4645, 0.4744, 0.4791, 0.4652, 0.4898, 0.4839, 0.4840, 0.4845, 0.4770, 0.4539),
  RF_Accuracy = c(0.7031, 0.7176, 0.7176, 0.7191, 0.7132, 0.7060, 0.7249, 0.7263, 0.7132, 0.7103, 0.7118),
  RF_MacroF1 = c(0.6292, 0.6491, 0.6497, 0.6466, 0.6512, 0.6405, 0.6584, 0.6601, 0.6404, 0.6387, 0.6401),
  RF_Kappa = c(0.4617, 0.4936, 0.4906, 0.4934, 0.4862, 0.4723, 0.5044, 0.5047, 0.4777, 0.4766, 0.4790))


# Pivot longer for easier plotting
plot_df <- pivot_longer(results_df, cols = -Top_Features, names_to = "Metric", values_to = "Score")

# Plot Accuracy
ggplot(subset(plot_df, grepl("Accuracy", Metric)), aes(x = Top_Features, y = Score, color = Metric)) +
  geom_line(size = 1.2) +
  geom_point(size = 2) +
  labs(title = "Accuracy Comparison: SVM vs Random Forest",
       x = "Number of Features",
       y = "Accuracy") +
  theme_minimal()

# Plot Macro F1
ggplot(subset(plot_df, grepl("MacroF1", Metric)), aes(x = Top_Features, y = Score, color = Metric)) +
  geom_line(size = 1.2) +
  geom_point(size = 2) +
  labs(title = "Macro F1 Score Comparison: SVM vs Random Forest",
       x = "Number of Features",
       y = "Macro F1 Score") +
  theme_minimal()

# Plot Kappa
ggplot(subset(plot_df, grepl("Kappa", Metric)), aes(x = Top_Features, y = Score, color = Metric)) +
  geom_line(size = 1.2) +
  geom_point(size = 2) +
  labs(title = "Kappa Score Comparison: SVM vs Random Forest",
       x = "Number of Features",
       y = "Kappa") +
  theme_minimal()



                        ###  Association Rules  ###

#  Prepare data — convert binary features to factors (YES/NO)
mine_features <- molecules[, !(colnames(molecules) %in% "Labels")]
mine_features <- as.data.frame(lapply(mine_features, function(x) {
  as.factor(ifelse(x == 1, "YES", "NO"))
}))

# Add the label column as strings like "Activity_Label=0"
mine_features$Activity_Label <- factor(molecules$Labels, levels = 0:2)

# Convert this complete dataframe to transactions
transactions <- as(mine_features, "transactions")

# Run apriori 
rules <- apriori(transactions,
                 parameter = list(supp = 0.01, conf = 0.6, minlen = 2, maxlen = 4),
                 appearance = list(
                   rhs = c("Activity_Label=0", "Activity_Label=1", "Activity_Label=2"),
                   default = "lhs")
)

# Show top 10 rules sorted by lift
inspect(sort(rules, by = "lift")[1:10])

# View top 10 most confident rules
inspect(sort(rules, by = "confidence", decreasing = TRUE)[1:10])
