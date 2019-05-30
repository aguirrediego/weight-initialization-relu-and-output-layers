from enum import Enum
from models.simple_cnn_model import SipleCNNModel
from models.vgg_16_model import VGG16Model
from models.inception_v3 import InceptionV3
from models.vgg_19_model import VGG19Model


class ModelOptions(Enum):
    SIMPLE_CNN = 1
    VGG16 = 2
    VGG19 = 3
    INCEPTIONV3 = 4


class ModelManager:
    @staticmethod
    def get_model(model_option, instance_shape, num_classes):

        model_dictionary = {
            ModelOptions.SIMPLE_CNN: SipleCNNModel,
            ModelOptions.VGG16: VGG16Model,
            ModelOptions.INCEPTIONV3: InceptionV3,
            ModelOptions.VGG19: VGG19Model

        }

        model_class = model_dictionary.get(model_option, SipleCNNModel)

        return model_class.get_model(instance_shape, num_classes)


def test_model_manager():
    model = ModelManager.get_model(ModelOptions.SIMPLE_CNN, (32, 32, 3), 10)

    print(model)


if __name__ == "__main__":
    test_model_manager()
