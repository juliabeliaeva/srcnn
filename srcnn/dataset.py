import tensorflow as tf


def resize(image, size, method):
    resized = tf.image.resize(image, size, method)
    if resized.dtype == tf.uint8:
        return resized
    else:
        # deal with https://github.com/tensorflow/tensorflow/issues/10722
        return tf.saturate_cast(resized, tf.uint8)


def prepare_input_image(original_image, input_h, input_w, scale):
    degraded_image = resize(original_image, size=[input_h // scale, input_w // scale],
                            method=tf.image.ResizeMethod.GAUSSIAN)
    input_image = resize(degraded_image, size=[input_h, input_w], method=tf.image.ResizeMethod.BICUBIC)
    return tf.image.convert_image_dtype(input_image, dtype=tf.float32)


def prepare_output_image(original_image, input_h, input_w, output_h, output_w):
    if output_h == input_h & output_w == input_w:
        output = original_image
    else:
        new_x = (input_w - output_w) // 2
        new_y = (input_h - output_h) // 2
        output = tf.image.crop_to_bounding_box(original_image, new_y, new_x, output_h, output_w)
    return tf.image.convert_image_dtype(output, dtype=tf.float32)


def prepare_crops(filename, input_size, stride, padding=None):
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_bmp(image_string)
    if padding is not None:
        image = tf.pad(image, tf.contsant([[padding, padding], [padding, padding], [0, 0]]), "CONSTANT")
    crops = tf.image.extract_patches(tf.expand_dims(image, 0), [1, input_size, input_size, 1],
                                     [1, stride, stride, 1], [1, 1, 1, 1], 'VALID')
    return tf.reshape(crops[0], [-1, input_size, input_size, 3])


def parse_function(filename, input_size, output_size, stride, scale):
    crops = prepare_crops(filename, input_size, stride)
    input_dataset = tf.data.Dataset.from_tensors(crops).map(
        lambda img: prepare_input_image(img, input_size, input_size, scale))
    output_dataset = tf.data.Dataset.from_tensors(crops).map(
        lambda img: prepare_output_image(img, input_size, input_size, output_size, output_size))
    result = tf.data.Dataset.zip((input_dataset, output_dataset))
    return result
