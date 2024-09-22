import tensorflow as tf
import larq as lq

# Create a simple Larq model
model = tf.keras.Sequential([
    lq.layers.QuantDense(32, input_shape=(10,), kernel_quantizer="ste_sign"),
    tf.keras.layers.Activation("relu"),
    lq.layers.QuantDense(10, kernel_quantizer="ste_sign")
])

# Compile the model
model.compile(optimizer="adam", loss="mse")

print("Model created successfully!")