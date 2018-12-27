# Sentiment-Analysis-on-Amazon-Customer-Reviews
Big Data Programming Group Project


Our project utilizes our knowledge and skills to process, analysis, and visualize data. Our team is going to use Amazon Customer Reviews dataset, which was conducted by Amazon over two decades since 1995 to collect Amazon customers’ opinions and experiences of using purchased products from Amazon.com website. This dataset is available to download from AWS (Amazon Web Services), and our team plan to focus customer reviews only on [electronic] category. 

We will use pyspark and sparksql as our main tools. Our task is to perform sentiment analysis on the reviews to find insights and relationships in reviews through the key words in body and understand how positive or negative the review is. We use review contents as features and star rating as target label to build our models and do prediction, which is a binary classification problem.

The dataset has 1.73GB with total 3093869 review records. 

Below are the features:

 |-- marketplace: string (nullable = true)
 |-- customer_id: string (nullable = true)
 |-- review_id: string (nullable = true)
 |-- product_id: string (nullable = true)
 |-- product_parent: string (nullable = true)
 |-- product_title: string (nullable = true)
 |-- product_category: string (nullable = true)
 |-- star_rating: string (nullable = true)
 |-- helpful_votes: string (nullable = true)
 |-- total_votes: string (nullable = true)
 |-- vine: string (nullable = true)
 |-- verified_purchase: string (nullable = true)
 |-- review_headline: string (nullable = true)
 |-- review_body: string (nullable = true)
 |-- review_date: string (nullable = true)
