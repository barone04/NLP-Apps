import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)


def main():
    # 1. Initialize Spark Session
    # .master("local[*]") runs Spark locally using all available CPU cores
    spark = SparkSession.builder \
        .appName("SentimentAnalysis") \
        .master("local[*]") \
        .getOrCreate()

    # Reduce Spark's verbose logging
    spark.sparkContext.setLogLevel("WARN")

    print("Spark Session initialized.")

    # 2. Load and Prepare Data
    data_path = os.path.join(project_root, 'data', 'sentiments.csv')

    try:
        df = spark.read.csv(data_path, header=True, inferSchema=True)
    except Exception as e:
        print(f"Error: Could not read data file at: {data_path}")
        print("Please ensure the file 'data/sentiments.csv' exists.")
        print(f"Error details: {e}")
        spark.stop()
        return

    # Convert -1/1 labels to 0/1 to suit Logistic Regression
    # (-1 + 1) / 2 = 0
    # ( 1 + 1) / 2 = 1
    df = df.withColumn("label", (col("sentiment").cast("integer") + 1) / 2)

    # Drop rows with null values
    initial_row_count = df.count()
    df = df.dropna(subset=["sentiment", "text", "label"])
    final_row_count = df.count()
    print(f"Loaded {initial_row_count} rows. Removed {initial_row_count - final_row_count} error/null rows.")
    print("Data after cleaning:")
    df.show(5)

    # Split data into training and test sets
    (trainingData, testData) = df.randomSplit([0.8, 0.2], seed=42)
    print(f"Training samples count: {trainingData.count()}")
    print(f"Test samples count: {testData.count()}")

    # 3. Build Preprocessing Pipeline
    # Stage 1: Split text into words (tokens)
    tokenizer = Tokenizer(inputCol="text", outputCol="words")

    # Stage 2: Remove stop words
    stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered_words")

    # Stage 3: Convert words to feature vectors (HashingTF)
    hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=10000)

    # Stage 4: Calculate IDF weights
    idf = IDF(inputCol="raw_features", outputCol="features")

    # 4. Train the Model
    # Stage 5: Logistic Regression model
    lr = LogisticRegression(
        maxIter=10,
        regParam=0.001,
        featuresCol="features",
        labelCol="label"
    )

    # Assemble all stages into a single Pipeline
    pipeline = Pipeline(stages=[tokenizer, stopwordsRemover, hashingTF, idf, lr])

    # Train the model
    print("\nStarting to train the Pipeline model...")
    model = pipeline.fit(trainingData)
    print("Training complete.")

    print("Evaluating the model on the test set...")
    predictions = model.transform(testData)

    print("A few prediction results:")
    predictions.select("text", "label", "prediction", "probability").show(10, truncate=50)


    # Accuracy
    evaluator_acc = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="accuracy"
    )
    accuracy = evaluator_acc.evaluate(predictions)

    # F1 Score
    evaluator_f1 = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="f1"
    )
    f1 = evaluator_f1.evaluate(predictions)

    print("\n=== Model Evaluation Results ===")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"F1-Score: {f1:.4f}")

    # Stop the Spark Session
    spark.stop()


if __name__ == "__main__":
    main()