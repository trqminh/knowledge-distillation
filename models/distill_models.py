import torch.nn as nn


class TeacherModel(nn.Module):
    def __init__(self, model):
        super(TeacherModel, self).__init__()
        self.model = model

    def forward(self, inputs):
        return self.model(inputs)


class StudentModel(nn.Module):
    def __init__(self, model):
        super(StudentModel, self).__init__()
        self.model = model

    def forward(self, inputs):
        return self.model(inputs)


class DistillationModel(nn.Module):
    def __init__(self, teacher_model, student_model):
        super(DistillationModel, self).__init__()
        self.teacher_model = teacher_model
        self.student_model = student_model

    def forward(self, inputs):
        return self.teacher_model(inputs), self.student_model(inputs)
