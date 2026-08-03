import traceback
from detection import Detector

print('starting')
try:
    d = Detector(model_type='yolov11', conf_threshold=0.3)
    print('loaded', type(d.model), d.model_type)
    print('names', len(d.names) if d.names is not None else None)
except Exception:
    traceback.print_exc()
