# bdp project 
# sentiment analysis for amazon reviews
# Amazon Review Text Mining and Sentiment Predictor

#################################### EDA ####################################

from pyspark.sql import SQLContext
df = sqlContext.read.format("com.databricks.spark.csv").option("delimiter", "\t").option("header", "true").load("amazon_reviews_us_Electronics_v1_00.tsv")

df.printSchema()

disParent = df.groupBy("marketplace").agg(countDistinct("product_parent"))
disParent.show()

df_RateProd = df.select(
        df.product_title,
        df.star_rating.cast("int")
    )

avgRatingForProd = df_RateProd.groupBy("product_title").mean("star_rating")   # not sort yet
avgRatingForProd.show()

df_select = df.select(
        df.helpful_votes.cast("int"),
        df.total_votes.cast("int"),
        df.vine,
        df.verified_purchase,
        df.review_headline,
        df.review_body,
        df.star_rating.cast("int")
    )

from pyspark.sql.functions import when, col
from pyspark.sql import functions as F

df_cvt1 = df_select.withColumn("vine",
    F.when(df_select["vine"]=="Y",1).
    otherwise(0))
df_cvt2 = df_cvt1.withColumn("verified_purchase",
    F.when(df_cvt1["verified_purchase"]=="Y",1).
    otherwise(0))
df = df_cvt2.withColumn("star_rating",
    F.when(df_cvt1["star_rating"]>=4,1).
    otherwise(0))

def votePerc(helpful_votes, total_votes):
    perc = (helpful_votes - (total_votes - helpful_votes)) / total_votes
    return perc

dfPerc = df.withColumn("vote_perc",
    F.when(df["total_votes"]==0,0).
    otherwise(votePerc(df["helpful_votes"], df["total_votes"])))

df_new = dfPerc.select(
    dfPerc.vine,
    dfPerc.verified_purchase,
    dfPerc.review_body,
    dfPerc.vote_perc,
    dfPerc.star_rating
)

from pyspark.sql.functions import countDistinct
ratingCount = df_new.groupBy("star_rating").count()

df_new = df_new.na.drop()
amazon = df_new
amazon.printSchema()

from pyspark.sql import SQLContext, Row
from pyspark.ml.feature import Tokenizer
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.feature import HashingTF, IDF
from pyspark.ml import Pipeline

# Tokenize the review
tokenizer = Tokenizer(inputCol = "review_body", outputCol = "review_words")
wordsDF = tokenizer.transform(amazon)

remover = StopWordsRemover(inputCol = "review_words", outputCol = "filtered")
wordsDF2 = remover.transform(wordsDF)
hashingTF = HashingTF(inputCol = "filtered", outputCol = "TF", numFeatures = 10000)
wordsDF3 = hashingTF.transform(wordsDF2)

idf = IDF(inputCol="TF", outputCol="features", minDocFreq = 5)   # minDocFreq: remove sparse terms
idfModel = idf.fit(wordsDF3)
wordsDF4 = idfModel.transform(wordsDF3)

# Split data into training and testing set
(training, test) = amazon.randomSplit([0.7, 0.3], seed = 100)


minor = training.where(col("star_rating") == 0)
countMinor = minor.groupBy("star_rating").count()

major = training.where(col("star_rating") == 1)
countMajor = major.groupBy("star_rating").count()

underSampling = major.sample(withReplacement = False, fraction = 0.33, seed = 100)
countUnderS = underSampling.groupBy("star_rating").count()
df_concat = minor.union(underSampling)
train = df_concat.withColumnRenamed("star_rating", "label")
countLabel = train.groupBy("label").count()


############# LRclf ################
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

lr = LogisticRegression(maxIter=20)

pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idfModel, lr])

paramGrid = ParamGridBuilder() \
    .addGrid(hashingTF.numFeatures, [10, 50]) \
    .addGrid(lr.regParam, [0.1, 0.01]) \
    .build()

crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=BinaryClassificationEvaluator(),
                          numFolds=4) 

cvModel = crossval.fit(train)

prediction = cvModel.transform(test)
selected = prediction.select("review_body", "star_rating", "probability", "prediction").take(5)
for row in selected:
    print(row)

evaluator = BinaryClassificationEvaluator(
    labelCol="star_rating")
Accuray = evaluator.evaluate(prediction)
print("Accuracy for Logistic Repression: " + str(Accuray))




############# RFclf ################
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

rf = RandomForestClassifier(numTrees=15)

pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idfModel, rf])

paramGrid = ParamGridBuilder() \
    .addGrid(hashingTF.numFeatures, [10, 50]) \
    .addGrid(rf.maxDepth, [5, 15]) \
    .build()

crossval = CrossValidator(estimator = pipeline,
                          estimatorParamMaps = paramGrid,
                          evaluator = BinaryClassificationEvaluator(),
                          numFolds = 5)

cvModel = crossval.fit(train)

prediction = cvModel.transform(test)
selected = prediction.select("review_body", "star_rating", "probability", "prediction").take(5)

for row in selected:
    print(row)

evaluator = BinaryClassificationEvaluator(
    labelCol="star_rating")
Accuracy_RF = evaluator.evaluate(prediction)

print("Accuracy for Random Forest: " + str(Accuracy_RF))






############# GBDTclf ################

from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier

# Train a GBT model.
gbt = GBTClassifier(maxIter=10)

# Chain indexers and GBT in a Pipeline
pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idfModel, gbt])

paramGrid = ParamGridBuilder() \
    .addGrid(hashingTF.numFeatures, [10, 50]) \
    .build()

crossval = CrossValidator(estimator = pipeline,
                          estimatorParamMaps = paramGrid,
                          evaluator = BinaryClassificationEvaluator(),
                          numFolds = 3)


# Train model.  This also runs the indexers.
cvModel = crossval.fit(train)

# Make predictions.
predictions = cvModel.transform(test)

# Select example rows to display.
selected_GBT = predictions.select("review_body", "star_rating", "probability", "prediction").take(5)

for row in selected_GBT:
    print(row)

evaluator_GBT = BinaryClassificationEvaluator(
    labelCol="star_rating")
Accuray_GBT = evaluator_GBT.evaluate(predictions)
print("Accuracy for Gradient-boosted tree classifier: " + str(Accuray_GBT))


