import cv2
import numpy as np
from PIL import Image
import math
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File

app = FastAPI()

model_path = 'android.tflite'
classes = ['rebar']
COLORS = np.random.randint(0, 255, size=(len(classes), 3), dtype=np.uint8)

def preprocess_image(image_path, input_size):
    img = tf.io.read_file(image_path)
    img = tf.io.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.uint8)
    original_image = img
    resized_img = tf.image.resize(img, input_size)
    resized_img = resized_img[tf.newaxis, :]
    resized_img = tf.cast(resized_img, dtype=tf.uint8)
    return resized_img, original_image

def detect_objects(interpreter, image, threshold):
    signature_fn = interpreter.get_signature_runner()
    output = signature_fn(images=image)
    count = int(np.squeeze(output['output_0']))
    scores = np.squeeze(output['output_1'])
    classes = np.squeeze(output['output_2'])
    boxes = np.squeeze(output['output_3'])
    results = []
    for i in range(count):
        if scores[i] >= threshold:
            result = {
                'bounding_box': boxes[i],
                'class_id': classes[i],
                'score': scores[i]
            }
            results.append(result)
    return results

def run_odt_and_draw_results(image_path, interpreter, threshold=0.25, iou_threshold=0.2):
    _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']
    preprocessed_image, original_image = preprocess_image(
        image_path,
        (input_height, input_width)
    )
    results = detect_objects(interpreter, preprocessed_image, threshold=threshold)
    boxes = np.array([result['bounding_box'] for result in results])
    scores = np.array([result['score'] for result in results])
    selected_indices = tf.image.non_max_suppression(
        boxes, scores, max_output_size=400, iou_threshold=iou_threshold
    ).numpy()
    selected_results = [results[i] for i in selected_indices]
    print(len(selected_results))
    original_image_np = original_image.numpy().astype(np.uint8)
    for obj in selected_results:
        ymin, xmin, ymax, xmax = obj['bounding_box']
        xmin = int(xmin * original_image_np.shape[1])
        xmax = int(xmax * original_image_np.shape[1])
        ymin = int(ymin * original_image_np.shape[0])
        ymax = int(ymax * original_image_np.shape[0])
        center_point = ((xmin + xmax) // 2, (ymin + ymax) // 2)
        width = abs(xmax - xmin)
        height = abs(ymax - ymin)
        radius = int(0.3 * math.sqrt(width**2 + height**2))
        cv2.circle(original_image_np, center_point, radius, (255,0,0), thickness=3)
    original_uint8 = original_image_np.astype(np.uint8)
    return original_uint8

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    with open(file.filename, 'wb') as buffer:
        buffer.write(file.file.read())
    detection_result_image = run_odt_and_draw_results(file.filename, interpreter)
    pil_image = Image.fromarray(detection_result_image)
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue())
    return HTMLResponse(content=f"<img src='data:image/png;base64,{img_str.decode()}'/>", status_code=200)
