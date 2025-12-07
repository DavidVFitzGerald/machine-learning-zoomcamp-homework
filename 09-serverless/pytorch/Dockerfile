FROM public.ecr.aws/lambda/python:3.13

RUN pip install onnxruntime keras-image-helper

ARG MODEL_NAME=clothing_classifier_mobilenet_v2.onnx
ENV MODEL_NAME=${MODEL_NAME}

COPY ${MODEL_NAME} ${MODEL_NAME}
COPY ${MODEL_NAME}.data ${MODEL_NAME}.data

COPY lambda_function.py .

CMD ["lambda_function.lambda_handler"]