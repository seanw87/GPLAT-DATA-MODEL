import numpy as np
import pandas as pd
import tensorflow as tf

data_dir = "dataset/"

def load_data():
    # list_file = data_dir + "ingame_analysis_list.csv"
    # csv_data = pd.read_csv(list_file, encoding="utf-8")
    # csv_data.describe().to_csv(data_dir + "csv_data_desc.csv")

    robot_csv_data = pd.read_csv(data_dir + "robot_1118.csv", encoding="utf-8")
    robot_csv_data.describe().to_csv(data_dir + "robot_1118_desc.csv")
    real_csv_data = pd.read_csv(data_dir + "real_1118.csv", encoding="utf-8")
    real_csv_data.describe().to_csv(data_dir + "real_1118_desc.csv")
    # print(robot_csv_data, real_csv_data)
    csv_data = pd.concat([robot_csv_data, real_csv_data], ignore_index = True)

    ret_csv_data = csv_data.drop("uid", axis=1).fillna(0).apply(np.float32)
    print(ret_csv_data)
    return ret_csv_data, csv_data

points, raw_data = load_data()
data_size = points.shape


num_points = data_size[0]
dimensions = data_size[1]

sess = tf.Session()
points_norm = tf.nn.l2_normalize(points, axis = 1)
points_norm_res = sess.run(points_norm)
np.savetxt(data_dir + "points_norm_res.csv", points_norm_res, delimiter=",")
# print(points_norm_res, type(points_norm_res))
# exit(0)

def input_fn():
    return tf.train.limit_epochs(tf.convert_to_tensor(points_norm_res, dtype=tf.float32), num_epochs=1)

num_clusters = 2
kmeans = tf.contrib.factorization.KMeansClustering(num_clusters=num_clusters, use_mini_batch=False)

#train
num_iterations = 15
previous_centers = None
for _ in range(num_iterations):
    kmeans.train(input_fn)
    cluster_centers = kmeans.cluster_centers()
    # if previous_centers is not None:
    #     print('delta:', cluster_centers - previous_centers)
    previous_centers = cluster_centers
    print('score:', kmeans.score(input_fn))
print('cluster centers:', cluster_centers)
np.savetxt(data_dir + "cluster_centers.csv", cluster_centers, delimiter=",")

#map the input points to their clusters
cluster_indices = list(kmeans.predict_cluster_index(input_fn))


def iter_row(row):
    index = row.name
    cluster_index = cluster_indices[index]
    # center = cluster_centers[cluster_index]
    # print('uid:', row['uid'], 'is in cluster', cluster_index, 'centered at', center)
    print('uid:', row['uid'], 'is in cluster', cluster_index)

raw_data.apply(lambda row: iter_row(row), axis=1)