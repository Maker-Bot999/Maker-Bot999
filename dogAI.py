import jetson.inference
import jetson.utils

net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
camera = jetson.utils.gstCamera(1280, 720, "/dev/video1")
display = jetson.utils.glDisplay()

while display.IsOpen():
  img, width, height = camera.CaptureRGBA()
  detections = net.Detect(img, width, height)
  display.RenderOnce(img, width, height)
  display.SetTitle("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))
  for detection in detections:
    class_name = net.GetClassDesc(detection.ClassID)
    name = f"Detected '{class_name}'"
   if "dog" in name:
    print("Yay! You found the hidden dog!")
    

