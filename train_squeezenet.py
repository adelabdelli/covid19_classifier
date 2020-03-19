import numpy
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Model
from keras_squeezenet import SqueezeNet
from keras.layers import GlobalAveragePooling2D, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import optimizers

NUM_CLASSES = 2
CHANNELS = 3
image_size = 224
DENSE_LAYER_ACTIVATION = 'softmax'
OBJECTIVE_FUNCTION = 'categorical_crossentropy'

LOSS_METRICS = ['accuracy']
NUM_EPOCHS = 100
EARLY_STOP_PATIENCE = 30

BATCH_SIZE_TRAINING = 16
BATCH_SIZE_VALIDATION = 16


base_model = SqueezeNet(include_top = False, weights = 'imagenet', input_shape = [image_size,image_size,3]) 
x = base_model.output
x = GlobalAveragePooling2D()(x)
preds = Dense(NUM_CLASSES,activation=DENSE_LAYER_ACTIVATION)(x) 
model = Model(inputs=base_model.input,outputs=preds)

model.summary()

opt = optimizers.Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.99, amsgrad=False)
model.compile(optimizer = opt, loss = OBJECTIVE_FUNCTION,  metrics =  LOSS_METRICS)

data_generator = ImageDataGenerator(rescale=1.0/255.0,
								   preprocessing_function=preprocess_input,                                  
                                   horizontal_flip=True,
                                   vertical_flip=True,
								   zoom_range=0.01,
								   shear_range=0.2,
                                   brightness_range=[0.2,1.0])
								   							   
data_generator2 = ImageDataGenerator(rescale=1.0/255.0,preprocessing_function=preprocess_input)                            

train_generator = data_generator.flow_from_directory(
        'trainset/',
        target_size=(image_size, image_size),
        batch_size=BATCH_SIZE_TRAINING,
        class_mode='categorical')

validation_generator = data_generator2.flow_from_directory(
        'testset/',
        target_size=(image_size, image_size),
        batch_size=BATCH_SIZE_VALIDATION,
        class_mode='categorical')

filepath="weights-improvement-{epoch:02d}-vacc:{val_accuracy:.2f}-tacc:{accuracy:.2f}.hdf5"
        
cb_early_stopper = EarlyStopping(monitor = 'val_loss', mode='min', verbose=1, patience = EARLY_STOP_PATIENCE)
cb_checkpointer = ModelCheckpoint(filepath = filepath, monitor = 'val_accuracy', save_best_only = False, mode = 'auto')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7)
       
fit_history = model.fit_generator(train_generator,
        epochs = NUM_EPOCHS,
        validation_data=validation_generator,
        verbose=2,      
        callbacks = [cb_checkpointer, cb_early_stopper, reduce_lr])
		
metrics = fit_history.history.keys()
t_acc = fit_history.history['accuracy']
t_loss = fit_history.history['loss']
v_acc  = fit_history.history['val_accuracy']
v_loss = fit_history.history['val_loss']

numpy.save('t_acc.npy',t_acc)
numpy.save('t_loss.npy',t_loss)
numpy.save('v_acc.npy',v_acc)
numpy.save('v_loss.npy',v_loss)