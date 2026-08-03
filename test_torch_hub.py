import torch
import traceback

print('torch version', torch.__version__)
print('hub module', getattr(torch, 'hub', None))
try:
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)
    print('model type', type(model))
    print('model', model)
    print('model names', getattr(model, 'names', None))
except Exception:
    traceback.print_exc()
