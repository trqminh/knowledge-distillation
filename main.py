from models import *
from train import *
from utils import *
import torch


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # data
    data_dict = process_data('./datasets/hymenoptera_data/', batch_size=4)
    # declare model
    teacher_model = ResNet(num_classes=2).to(device)
    # teacher_model.load_state_dict(torch.load('./train/teacher_model.pth'))
    student_model = SimpleNeuralNet().to(device)
    student_distilled_model = SimpleNeuralNet().to(device)

    # train
    teacher_acc = train_single_model(data_dict, teacher_model, device, 'teacher', num_epochs=10)
    student_acc = train_single_model(data_dict, student_model, device, 'student', num_epochs=10)
    student_distill_acc = train_student_model(data_dict, student_distilled_model, teacher_model, device,
                                              alpha=0.5, num_epochs=10)

    print('*' * 10)
    print('Teacher model acc: ', teacher_acc)
    print('Student model acc: ', student_acc)
    print('Distillation student model from teacher model: ', student_distill_acc)


if __name__ == '__main__':
    main()
