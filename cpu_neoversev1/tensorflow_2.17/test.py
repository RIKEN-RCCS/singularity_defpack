import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

tensor = tf.constant([1.0, 2.0, 3.0])
print("Tensor:", tensor)

result = tf.reduce_sum(tensor)
print("Sum of Tensor:", result)
