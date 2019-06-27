# Run Trained Model
#### To run the pretrained model by loading the parameters use the [main.py](/main.py) file.

# Train Model
### [Seq2SeqModel.py](/Seq2SeqModel.py) trains the model by importing and calling the required predefined functions.
#### The sequence of python files to be followed:
1. [Data_preprocessing.py](/Data_preprocessing.py) : Source and Target words are converted into integer sequence.
2. [Model_Inputs.py](/Model_Inputs.py): Placeholders are defined here.
3. [Layers.py](/Layers.py): Encoder and Decoder layers are defined here.
4. [Batch_Metrics.py](/Batch_Metrics.py): Batch generation for training.
