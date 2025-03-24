# LSeg: trimmed down fork for use with EDI-SLAMVDB

Forked from: https://github.com/isl-org/lang-seg/tree/65c0a0978fa2e562cda695afed1554c63ec35cb

License: redistributed under MIT, added copyright notice for changes made after fork. Only the code necessary for inference has been retained.

Run ` pip install -r requirements.txt ` and download LSeg weights from [HERE](https://pubfaili.edi.lv/s/BPsG7PDiijtj9CA) (place the weights in the folder named weights) to get started. 

## Inference

The intended usage of this package is through the *segmentation_service.py* script, which exposes image segmentation and text embedding APIs as ROS2 services. Refer to the main project for more details.