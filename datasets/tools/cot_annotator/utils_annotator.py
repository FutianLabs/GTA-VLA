# 1. Initialize the client with your API token.
from dds_cloudapi_sdk import Config
from dds_cloudapi_sdk import Client
# 2. Create a task with proper parameters.
from dds_cloudapi_sdk.tasks.v2_task import V2Task, create_task_with_local_image_auto_resize


def get_dds_client(token):
    config = Config(token)
    client = Client(config)
    return client


def get_detection_result(image_path, text_prompt, dds_client, bbox_threshold=0.1):
    api_path = "/v2/task/dinox/detection"
    api_body = {
        "model": "DINO-X-1.0",
        "prompt": {
            "type":"text",
            "text":text_prompt
        },
        "targets": ["bbox", "mask"],
        "bbox_threshold": bbox_threshold,
        "iou_threshold": 0.8,
        "mask_format": "coco_rle",
    }

    task = create_task_with_local_image_auto_resize(
        api_path=api_path,
        api_body_without_image=api_body,
        image_path=image_path,
    )
    dds_client.run_task(task)

    return task.result


