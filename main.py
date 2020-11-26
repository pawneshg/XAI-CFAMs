from model.resnet18 import resnet18, test_resnet18

model = resnet18()
loss, acc = test_resnet18(model)
print("Model Test Acc", acc)
print("Model Test Loss", loss)