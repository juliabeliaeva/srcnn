import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from IPython.display import display, HTML


def plot_image(image, label=''):
    plt.imshow(image)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(label)


def plot_images(images, grid_size):
    plt.figure(figsize=(8, 8))
    for n, image in enumerate(images[0:grid_size ** 2]):
        plt.subplot(grid_size, grid_size, n + 1)
        plot_image(image, n)


def plot_dataset(d, n_patches=8):
    plt.figure(figsize=(2 * n_patches, 8))
    for n, images in enumerate(d.take(n_patches)):
        input_image = images[0]
        output_image = images[1]
        ih, iw, _ = input_image.get_shape()
        oh, ow, _ = output_image.get_shape()

        padding = (ih - oh) // 2
        padded_output = tf.pad(output_image, [[padding, padding], [padding, padding], [0, 0]], "CONSTANT")

        plt.subplot(n_patches, 2, 2 * n + 1)
        plot_image(tf.image.convert_image_dtype(input_image, dtype=tf.uint8), str(n))
        plt.subplot(n_patches, 2, 2 * n + 2)
        plot_image(tf.image.convert_image_dtype(padded_output, dtype=tf.uint8), str(n))


def plot_predictions(model, names, inputs, ground_truth):
    inputs = tf.stack(inputs)
    ground_truth = tf.stack(ground_truth)
    outputs = tf.clip_by_value(model.predict(inputs), 0, 1)

    _, in_h, in_w, _ = inputs.get_shape()
    _, out_h, out_w, _ = outputs.shape
    new_x = (in_w - out_w) // 2
    new_y = (in_h - out_h) // 2
    inputs = tf.image.crop_to_bounding_box(inputs, new_y, new_x, out_h, out_w)

    for i, o, g in zip(inputs, outputs, ground_truth):
        plt.figure(figsize=(12, 12))
        plt.subplot(1, 3, 1)
        plot_image(i, "input")
        plt.subplot(1, 3, 2)
        plot_image(o, "output")
        plt.subplot(1, 3, 3)
        plot_image(g, "ground truth")

    psnr_inputs = tf.image.psnr(inputs, ground_truth, max_val=1.0)
    psnr_outputs = tf.image.psnr(outputs, ground_truth, max_val=1.0)
    psnr = []
    for name, psnr_i, psnr_o in zip(names, psnr_inputs, psnr_outputs):
        psnr.append([name, f"{psnr_i:.3f}", f"{psnr_o:.3f}"])

    return psnr


def display_psnr(psnr):
    display(pd.DataFrame(psnr, columns=["Image Name", "PSNR: Input", "PSNR: SRCNN"]))