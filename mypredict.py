from ultralytics import YOLO

model = YOLO(r"yolo26x.pt")
model.predict(
    source=r"ultralytics\assets",
    save=True,
    show=False,
)
