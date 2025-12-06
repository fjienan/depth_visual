from ultralytics.models.yolo.pose import PoseTrainer
data_path = "/home/jienan/depth-visual/data_base/data_2025_12_5_1/data.yaml"
output_path = "/home/jienan/depth-visual/model"
# 不用与训练模型，从0开始训练
model = "/home/jienan/depth-visual/model/yolov8n-pose2/weights/best.pt"
args = dict(model=model, data=data_path, epochs=500, project=output_path, name="yolov8n-pose")
trainer = PoseTrainer(overrides=args)
trainer.train()