import tensorflow as tf
import time

class MatMulBenchmark(tf.test.Benchmark):
    def benchmark_matmul(self):
        x = tf.random.normal([1000, 1000])
        y = tf.random.normal([1000, 1000])
        start = time.time()
        for _ in range(100):
            z = tf.matmul(x, y)
        end = time.time()
        print(f"Execution time: {end - start:.5f} seconds")

bench = MatMulBenchmark()
bench.benchmark_matmul()
