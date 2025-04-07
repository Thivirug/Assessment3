# References used:
# https://arxiv.org/abs/1505.04597, 
# https://medium.com/geekculture/u-net-implementation-from-scratch-using-tensorflow-b4342266e406
# https://www.tensorflow.org/tutorials/images/segmentation
# https://www.tensorflow.org/guide/keras/serialization_and_saving#config_methods

from keras.layers import Conv2D, Activation, MaxPool2D, Concatenate, Conv2DTranspose  # type: ignore
from keras import Model, Input # type: ignore
import tensorflow as tf # type: ignore
import keras # type: ignore

@keras.saving.register_keras_serializable(package="my_models", name="UNet")
class UNet(Model):
    """
        Class that implements the UNet architecture. (With some modifications)
    """

    def __init__(self, input_shape: tuple[int, int, int], n_output_classes: int = 1, n_base_filters: int = 64, **kwargs) -> None:
        """
        Initialise UNet architecture.

        Args:
            n_output_classes: Number of output classes.
            input_shape: Input image shape (height, width, channels). Default is 1.
            n_base_filter: Number of convolutional filters in the first downsampling and upsampling layers. Default is 64.
            **kwargs: Additional arguments for the base class.
        """

        super().__init__(**kwargs)  

        # store class attributes
        self.input_shape = input_shape
        self.n_classes = n_output_classes
        self.n_filters = n_base_filters

        # store activation and Conv2D kernel initializer
        self.activation = 'relu'
        self.kernel_init = 'HeNormal'

        # store model
        self._model = self.build_model()

    def get_config(self):
        """
            Return a dictionary containing the constructor arguments to enable model reconstruction.
        """
        # get base config
        config = super().get_config()

        config.update({
            'input_shape': self.input_shape,
            'n_output_classes': self.n_classes,
            'n_base_filters': self.n_filters,
        })

        return config
    
    @classmethod
    def from_config(cls, config):
        """
            Create a new instance of the class from the config dictionary.
        """

        # extract input_shape, n_output_classes, n_base_filters
        input_shape = config.pop('input_shape')
        n_output_classes = config.pop('n_output_classes')
        n_base_filters = config.pop('n_base_filters')

        # create new instance
        return cls(
            input_shape = input_shape,
            n_output_classes = n_output_classes,
            n_base_filters = n_base_filters,
            **config
        )

    def downsampler_block(self, inputs: tf.Tensor, n_filters: int, kernel_size: int = 3) -> tf.Tensor:
        """     
            Single encoder block with Double convolutional layers & batch normalisation layers.

            Args:
                inputs: Tensor output of previous layer.
                n_filters: Number of convolutional filters.
                kernel_size: Height and width of the kernel/filter. Default is 3.
        """

        # pass inputs through the layers
        x = Conv2D(
            filters = n_filters,
            kernel_size = kernel_size,
            padding = 'same',
            kernel_initializer = self.kernel_init
        )(inputs)
        x = Activation(self.activation)(x)

        x = Conv2D(
            filters = n_filters,
            kernel_size = kernel_size,
            padding = 'same',
            kernel_initializer = self.kernel_init
        )(x)
        x = Activation(self.activation)(x)

        return x

    def upsampler_block(self, inputs: tf.Tensor, n_filters: int, skip_connection: tf.Tensor, kernel_size: int = 2) -> tf.Tensor:  
        """
            Single Decoder block with Conv2DTranspose layers, Conv2D layers, and Concatenate.

            Args:
                inputs: Tensor output of previous layer.
                n_filters: Number of convolutional filters.
                skip_connection: Output of the similar level Conv layer from the downsampling section.
                kernel_size: Height and width of the kernel/filter. Default is 2.
        """

        up_conv = Conv2DTranspose(
            filters = n_filters,
            kernel_size = kernel_size,
            strides = 2,
            padding = 'same'
        )(inputs)

        concat = Concatenate()(
            [up_conv, skip_connection]
        )

        x = Conv2D(
            filters = n_filters,
            kernel_size = kernel_size,
            padding = 'same',
            kernel_initializer = self.kernel_init,
            activation = self.activation
        )(concat)

        x = Conv2D(
            filters = n_filters,
            kernel_size = kernel_size,
            padding = 'same',
            kernel_initializer = self.kernel_init, 
            activation = self.activation
        )(x)

        return x
    
    def build_model(self) -> tf.keras.Model:
        """
            Build the entire UNet architecture.
        """

        inputs = Input(shape = self.input_shape)

        # ENCODER (Downsampling)
        d1 = self.downsampler_block(inputs, self.n_filters)
        mp1 = MaxPool2D(pool_size = (2,2))(d1)

        d2 = self.downsampler_block(mp1, 2*self.n_filters)
        mp2 = MaxPool2D(pool_size = (2,2))(d2)

        d3 = self.downsampler_block(mp2, 4*self.n_filters)
        mp3 = MaxPool2D(pool_size = (2,2))(d3)

        d4 = self.downsampler_block(mp3, 8*self.n_filters)
        mp4 = MaxPool2D(pool_size = (2,2))(d4)

        # BOTTLENECK
        bottleneck = self.downsampler_block(mp4, 16*self.n_filters)

        # DECODER (Upsampling)
        u4 = self.upsampler_block(bottleneck, 8*self.n_filters, d4)
        u3 = self.upsampler_block(u4, 4*self.n_filters, d3)
        u2 = self.upsampler_block(u3, 2*self.n_filters, d2)
        u1 = self.upsampler_block(u2, self.n_filters, d1)

        outputs = Conv2D(
            filters = self.n_classes,
            kernel_size = 1,
            padding = 'same',
            activation = 'sigmoid'
        )(u1)

        # create & return model
        return Model(
            inputs = inputs,
            outputs = outputs
        )


    def call(self, inputs: tf.Tensor):
        """
            Forward pass.

            Args:
                inputs: Image input tensors.
        
        """

        return self._model(inputs)