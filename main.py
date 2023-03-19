import os
from datetime import datetime, timedelta


import pyspark
from pyspark.sql import SparkSession
from pyspark.context import SparkContext
import pyspark.sql.functions as F
from pyspark.sql.types import DateType
from pyspark.sql.window import Window

spark = SparkSession.builder.master("local").getOrCreate()
path_data = "./data"
path_geo = "./geo"


def event_with_city(path_event_prqt: str, path_city_data: str,
                    spark: pyspark.sql.SparkSession) -> pyspark.sql.DataFrame:
    events_geo = spark.read.parquet(path_event_prqt) \
        .sample(0.05) \
        .withColumnRenamed('lat', 'msg_lat') \
        .withColumnRenamed('lon', 'msg_lon') \
        .withColumn('event_id', F.monotonically_increasing_id())

    city = spark.read.csv(path_city_data, sep=";", header=True)

    events_city = events_geo \
        .crossJoin(city) \
        .withColumn('diff', F.acos(
        F.sin(F.col('msg_lat')) * F.sin(F.col('lat')) + F.cos(F.col('msg_lat')) * F.cos(F.col('lat')) * F.cos(
            F.col('msg_lon') - F.col('lng'))) * F.lit(6371)).persist()

    return events_city


def event_corr_city(events_city: pyspark.sql.DataFrame, spark: pyspark.sql.SparkSession):
    window = Window().partitionBy('event.message_from').orderBy(F.col('diff').desc())

    df_city = events_city \
        .withColumn("row_number", F.row_number().over(window)) \
        .filter(F.col('row_number') == 1) \
        .drop('row_number')

    return df_city


def actial_geo(df_city: pyspark.sql.DataFrame, spark: pyspark.sql.SparkSession) -> pyspark.sql.DataFrame:
    window = Window().partitionBy('event.message_from').orderBy(F.col('date').desc())

    df_actual = df_city \
        .withColumn("row_number", F.row_number().over(window)) \
        .filter(F.col('row_number') == 1) \
        .selectExpr('event.message_from as user', 'city', 'id as city_id') \
        .persist()

    return df_actual


def travel_geo(df_city: pyspark.sql.DataFrame, spark: pyspark.sql.SparkSession) -> pyspark.sql.DataFrame:
    window = Window().partitionBy('event.message_from', 'id').orderBy(F.col('date'))

    df_travel = df_city \
        .withColumn("dense_rank", F.dense_rank().over(window)) \
        .withColumn("date_diff",
                    F.datediff(F.col('date').cast(DateType()), F.to_date(F.col("dense_rank").cast("string"), 'd'))) \
    .selectExpr('date_diff', 'event.message_from as user', 'date', "id") \
        .groupBy("user", "date_diff", "id") \
        .agg(F.countDistinct(F.col('date')).alias('cnt_city'))

    return df_travel


def home_geo(df_travel: pyspark.sql.DataFrame, spark: pyspark.sql.SparkSession) -> pyspark.sql.DataFrame:
    df_home = df_travel \
        .withColumn('max_dt', F.max(F.col('date_diff')) \
                    .over(Window().partitionBy('user'))) \
        .filter((F.col('cnt_city') > 27) & (F.col('date_diff') == F.col('max_dt'))) \
        .persist()

    return df_home

df_events_city = event_with_city(path_data, path_geo, spark)

df_city = event_corr_city(df_events_city, spark)
#df_actual = actial_geo(df_city, spark)
df_travel = travel_geo(df_city, spark)

#events_geo = spark.read.parquet("./data").withColumn('event_id', F.monotonically_increasing_id())
#df_actual.show(15, False)
df_travel.show(10, False)
home_geo(df_travel, spark).show(10)
