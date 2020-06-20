from darkflow.net.build import TFNet

options = {
	"model": "../Soccer-Ball-Detection-YOLOv2-master/cfg/yolo_custom.cfg",
	"batch": 8,
	"epoch": 100,
	"gpu": 1.0,
	"train": True,
	"annotation": "../data/annotations/",
	"dataset": "../data/dataset/"
}

tfnet = TFNet(options)
tfnet.train()
