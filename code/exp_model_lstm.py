"""
Note that this code is not executable as is. It is meant to be used as a reference for the CNN-LSTM model used in the project. Should you wish to run the code, you will need to modify the paths to the data and the train/validation split. As well as import the necessary libraries. The most important part of  thecode is the samples_size variable. This is the number of time steps in the image sequence. It is set to 41, which covers the time steps for every 6 minutes for 4 hours.

"""

class CustomGenerator(Sequence):
    
    """
    Custom generator for the LSTM model. 
    ---

    Parameters:
    directory: str
        The directory where the images are stored.
    batch_size: int
        The number of samples in the batch.

    samples_size: int
        The number of time steps in the image sequence.

    segment: str
        The train/validation split of the data to be used.
    ---
    Returns:
    x: numpy array
        The input image sequence.
    time_input: numpy array
        The input time sequence.
    output: numpy array
        The output position sequence.
    """
    def __init__(self, directory, samples_size=41, batch_size=32, segment='Train'):
        self.directory = directory
        self.samples_size = samples_size
        self.batch_size = batch_size
        self.segment = segment
        if self.segment != 'Train':
            self.files = files_val
        else:
            self.files = files_train
        self.idx = 0

    def __len__(self):
      # determines the number of batches that can be produced 
        return int(np.floor(len(self.files) / (self.batch_size * self.samples_size)))

    def __getitem__(self, idx):
        # generates one batch of data
        start = idx * self.batch_size * self.samples_size
        end = (idx + 1) * self.batch_size * self.samples_size
        batch = self.files[start:end]
        # create empty arrays to hold images, times and positions
        x = np.empty((self.batch_size, self.samples_size, 224, 224, 1))
        y = np.zeros((self.batch_size, self.samples_size, 2))
        times_gen = np.empty((self.batch_size, self.samples_size, 1))
        # loop through batch
        for i in range(self.batch_size):
            for j, img in enumerate(batch[i*self.samples_size:(i+1)*self.samples_size]):
                img_path = os.path.join(self.directory, img)
                # populate time and position labels 
                y[i, j] = extract_position_time(img_path)[0]
                time = extract_position_time(img_path)[1]
                times_gen[i, j] = normalize_datetime(time, times.min(), times.max())
                # read image
                img = cv2.imread(img_path, 0)
                img = cv2.resize(img, (224, 224))
                # add image to x
                x[i, j] = img.reshape(224, 224, 1)
        # get output
        output = normalize_y(y, position)
        # get time input
        time_input = times_gen
        return [x/255, time_input], output


# define the input layers
samples_size = 41 # this covers the time steps for ever 6 minutes for 4 hours
input_image = Input(shape=(samples_size, 224,224,1))
input_time = Input(shape=(samples_size, 1))

# process the input image using a CNN
x = TimeDistributed(Conv2D(filters=16, kernel_size=3, padding='same', activation='relu'))(input_image)
x = TimeDistributed(MaxPooling2D())(x)
x = TimeDistributed(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))(x)
x = TimeDistributed(MaxPooling2D())(x)
x = TimeDistributed(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))(x)
x = TimeDistributed(Flatten())(x)

# flatten the input time
t = TimeDistributed(Flatten())(input_time)

# concatenate the CNN and LSTM outputs
z = concatenate([x, t])

# pass to LSTM
z = LSTM(64, activation='relu', return_sequences=True)(z)

# z = concatenate([x, t])
z = Dense(64, activation='relu')(z)
z = Dropout(0.2)(z)
outputs = Dense(2, activation='sigmoid')(z)
print(outputs)

# define the model
experimental_model = Model(inputs=[input_image, input_time], outputs=outputs)
experimental_model.summary()