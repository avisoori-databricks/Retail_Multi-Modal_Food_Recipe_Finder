# Databricks notebook source
# MAGIC %md
# MAGIC ### Step 0. Installing Libraries and Loading the Data

# COMMAND ----------

# MAGIC %md
# MAGIC Install and import the necessary libraries

# COMMAND ----------

# MAGIC %pip install -U sentence-transformers 

# COMMAND ----------

import os, os.path
import glob
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import pyspark.pandas as ps
import torch
import pickle
from PIL import Image

# COMMAND ----------

# MAGIC %md 
# MAGIC Load the zipped image subdirectory (within which all images are contained) to dbfs using the file upload utility: https://www.kaggle.com/datasets/pes12017000148/food-ingredients-and-recipe-dataset-with-images. Uncomment the below code block to execute

# COMMAND ----------

#%sh
#unzip  /dbfs/FileStore/shared_uploads/avinash.sooriyarachchi@databricks.com/food_images.zip

# COMMAND ----------

# MAGIC %md
# MAGIC Make sure that the folder was unzipped properly

# COMMAND ----------

#%sh 
#ls

# COMMAND ----------

# MAGIC %md
# MAGIC The unzipped folder resides in /tmp. Move it to dbfs. Alternatively, move it to mounted cloud storage

# COMMAND ----------

#dbutils.fs.mv("file:/databricks/driver/food_images", "dbfs:/FileStore/shared_uploads/avinash.sooriyarachchi@databricks.com/foodImages", recurse=True)  

# COMMAND ----------

# MAGIC %md
# MAGIC Ensure the number of files in the directory is correct

# COMMAND ----------

directory = "/dbfs/FileStore/shared_uploads/avinash.sooriyarachchi@databricks.com/foodImages"
number_of_files = len(os.listdir(directory))
print(number_of_files)

# COMMAND ----------

# MAGIC %md
# MAGIC Upload the recipes.csv file (containing the recipes, corresponding images, ingredients etc. downloaded before) to a location in mounted storage or dbfs.
# MAGIC Read this into a pandas dataframe and read the image file names into a list (converting them to file locations in the process)

# COMMAND ----------

recipes = pd.read_csv("/dbfs/FileStore/shared_uploads/avinash.sooriyarachchi@databricks.com/food_images/food_recipes.csv")
recipe_image_names = recipes.Image_Name.to_list()
indexable_images = [directory+'/'+ str(name)+'.jpg' for name in recipe_image_names]

# COMMAND ----------

# MAGIC %md
# MAGIC Find which of the images in the previously unzipped images folder have corresponding recipes in the recipes dataframe. We want a 1:1 relationship here

# COMMAND ----------

img_names = list(glob.glob(directory+'/*.jpg'))

# COMMAND ----------

#get the actual indexable images that are in the images folder
final_index_images = []
for image in indexable_images:
  if image in img_names:
    final_index_images.append(image)
len(final_index_images)

# COMMAND ----------

# MAGIC %md
# MAGIC Save image paths into a delta table. Create a database and a table using the delta files for convenient querying

# COMMAND ----------

spark.createDataFrame(pd.DataFrame(final_index_images, columns = ['Path'])).write.format('delta').save('/tmp/image_paths')

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE DATABASE IF NOT EXISTS nrf_food_app;
# MAGIC CREATE TABLE IF NOT EXISTS nrf_food_app.image_paths USING DELTA LOCATION '/tmp/image_paths';

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Step 1. (Skip this) Computing embeddings from scratch (pickled embedding file is provided following this section)
# MAGIC The cells below show how to compute the embeddings in a distributed manner and build an index during the run of this notebook

# COMMAND ----------

#Load the ViT-Clip model
model = SentenceTransformer('clip-ViT-B-32')


# COMMAND ----------

#Define function for embedding calculation
def get_embeddings(img_loc):
  return model_new.encode(Image.open(img_loc).convert('RGB'), batch_size=128, convert_to_tensor=False, show_progress_bar=False)

# COMMAND ----------

df = spark.sql("SELECT * FROM nrf_food_app.image_paths")
display(df)

# COMMAND ----------

#Compute embeddings in a distributed manner using the pandas on pyspark API
ps.set_option('compute.default_index_type', 'distributed')
df_ = df.to_pandas_on_spark()
df_['embeddings'] = df_['Path'].apply(get_embeddings)

# COMMAND ----------

#Save the embeddings to delta and create table in database
df_sp = df_.to_spark()
df_sp.write.format('delta').save('/tmp/image_embeddings')

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS nrf_food_app.image_embeddings USING DELTA LOCATION '/tmp/image_embeddings';

# COMMAND ----------

#Convert embeddings to a format amnenable for indexing with sentence transformers

# COMMAND ----------

df_pd = df_.to_pandas()
np_array_list = df_pd.embeddings.to_list()
img_emb = torch.tensor(np_array_list)

# COMMAND ----------

# Next, we define a search function.
def search(query, k=3):
    # First, we encode the query (which can either be an image or a text string)
    query_emb = model.encode([query], convert_to_tensor=True, show_progress_bar=False)
    
    # Then, we use the util.semantic_search function, which computes the cosine-similarity
    # between the query embedding and all image embeddings.
    # It then returns the top_k highest ranked images, which we output
    hits = util.semantic_search(query_emb, img_emb, top_k=k)[0]
    
    print("Query:")

    return [final_index_images[hit['corpus_id']] for hit in hits]

# COMMAND ----------

# MAGIC %md
# MAGIC Save the image paths and corresponding image embeddings as pickle files for ease of reproducibility

# COMMAND ----------

picke_path = "/dbfs/FileStore/shared_uploads/avinash.sooriyarachchi@databricks.com/pickled_paths_embeddings.pkl"
with open(picke_path, "wb") as fOut:
    pickle.dump({'image_paths':final_index_images ,'embeddings': img_emb},fOut)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Step 2. (Skip To this) Load the paths and image embeddings saved as a pickle file for reproducibility

# COMMAND ----------

pickle_location = "/dbfs/FileStore/shared_uploads/avinash.sooriyarachchi@databricks.com/pickled_paths_embeddings.pkl"

# COMMAND ----------

unpickled = pickle.load(open(pickle_location, "rb"))

# COMMAND ----------

embeddings = unpickled['embeddings']
paths = unpickled['image_paths']

# COMMAND ----------

len(paths), len(embeddings)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3. Define function to lookup query string or image (multi-modal model) in the indexed embeddings
# MAGIC And return the corresponding image paths which can be rendered

# COMMAND ----------

# Next, we define a search function.
def search(query, k=3):
    # First, we encode the query (which can either be an image or a text string)
    query_emb = model.encode([query], convert_to_tensor=True, show_progress_bar=False)
    
    # Then, we use the util.semantic_search function, which computes the cosine-similarity
    # between the query embedding and all image embeddings.
    # It then returns the top_k highest ranked images, which we output
    hits = util.semantic_search(query_emb, embeddings, top_k=k)[0]
    
    print("Query:")

    return [paths[hit['corpus_id']] for hit in hits]

# COMMAND ----------

results = search("/dbfs/FileStore/shared_uploads/avinash.sooriyarachchi@databricks.com/lasagna1.png")
#results = search('nutella grilled cheese')

# COMMAND ----------

renders = []
for result in results:
  im = Image.open(result).convert('RGB')
  renders.append(im)
display(renders[0], renders[1], renders[2])

# COMMAND ----------


