import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)


def main():
    # 1. Initialize Spark Session
    spark = SparkSession.builder \
        .appName("NaiveBayesSentimentImprovement") \
        .master("local[*]") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    print("Spark Session initialized.")

    # 2. Load and prepare data (using -1/1 labels)
    data_path = os.path.join(project_root, 'data', 'sentiments.csv')

    try:
        df = spark.read.csv(data_path, header=True, inferSchema=True)
    except Exception as e:
        print(f"Error: Could not read data file at: {data_path}. Ensure the file exists.")
        spark.stop()
        return

    # Convert -1 (Negative) and 1 (Positive) labels to 0 and 1
    # Formula: (-1 + 1)/2 = 0; (1 + 1)/2 = 1
    df = df.withColumn("label", (col("sentiment").cast("integer") + 1) / 2)
    df = df.dropna(subset=["sentiment", "text", "label"])

    (trainingData, testData) = df.randomSplit([0.8, 0.2], seed=42)
    print(f"Training samples count: {trainingData.count()}")
    print(f"Test samples count: {testData.count()}")

    # 3. Build Preprocessing Pipeline (TF-IDF Feature Engineering)
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    # Setting numFeatures relatively high for better representation
    hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=10000)
    idf = IDF(inputCol="raw_features", outputCol="features")

    # 4. Model Replacement: Naive Bayes
    # Using multinomial model type suitable for count/frequency features like TF-IDF
    nb = NaiveBayes(
        smoothing=1.0,  # Smoothing parameter to handle zero counts
        featuresCol="features",
        labelCol="label",
        modelType="multinomial"
    )

    # Assemble the Pipeline
    pipeline = Pipeline(stages=[tokenizer, stopwordsRemover, hashingTF, idf, nb])

    # Train the model
    print("\nStarting to train the Naive Bayes Pipeline model...")
    model = pipeline.fit(trainingData)
    print("Training complete.")

    # 5. Evaluate the Model
    predictions = model.transform(testData)

    # Calculate Accuracy
    evaluator_acc = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="accuracy"
    )
    accuracy = evaluator_acc.evaluate(predictions)

    # Calculate F1 Score
    evaluator_f1 = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="f1"
    )
    f1 = evaluator_f1.evaluate(predictions)

    print("\n--- NAIVE BAYES MODEL EVALUATION RESULTS ---")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"F1-Score: {f1:.4f}")

    spark.stop()


if __name__ == "__main__":
    main()