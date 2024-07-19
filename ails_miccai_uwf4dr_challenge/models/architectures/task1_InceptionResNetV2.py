from keras.applications import InceptionResNetV2
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D


base_model = InceptionResNetV2(
    include_top=False,
    weights='imagenet',
    input_tensor=None,
    input_shape=(800, 1016, 3),
    pooling=avg,
    classes=1,
    classifier_activation='sigmoid'
)

#keras.applications.inception_resnet_v2.preprocess_input use with dataset




class InceptionResNetV2Model:
    def __init__(self, input_shape=(800, 1016, 3), num_classes=1, include_top=False, weights='imagenet'):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.include_top = include_top
        self.weights = weights
        self.model = self.build_model()
    
    def build_model(self):
        # Load the InceptionResNetV2 model
        base_model = InceptionResNetV2(
            include_top=self.include_top,
            weights=self.weights,
            input_shape=self.input_shape
        )
        
        # Add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # Add a fully-connected layer
        x = Dense(1024, activation='relu')(x)
        # Add the final output layer
        predictions = Dense(self.num_classes, activation='sigmoid')(x)
        
        # Create the model
        model = Model(inputs=base_model.input, outputs=predictions)
        
        return model
    
    def compile_model(self, learning_rate=0.001):
        # Compile the model
        self.model.compile(optimizer=RMSprop(learning_rate=learning_rate), loss='binary_crossentropy')
    
    def train_model(self, train_data, train_labels, epochs=10, batch_size=32, validation_data=None):
        # Train the model
        self.model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size, validation_data=validation_data)
    
    def freeze_base_model(self):
        # Freeze the base model layers
        for layer in self.model.layers[:-3]:  # Excluding the newly added layers
            layer.trainable = False
    
    def unfreeze_base_model(self):
        # Unfreeze all layers
        for layer in self.model.layers:
            layer.trainable = True
