# Run Trained Model
#### To run the pretrained model by loading the parameters use the `main.py` file.

# Train Model
### `Seq2SeqModel.py` trains the model by importing and calling the required predefined functions.
#### for reference the following order is to be followed:
1. `Data_preprocessing.py` : Source and Target words are converted into integer sequence.
2. `Model_Inputs.py`: Placeholders are defined here.
3. `Layers.py`: Encoder and Decoder layers are defined here.
4. `Batch_Metrics.py`: Batch generation for training.
