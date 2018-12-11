To install necessary libraries:

$ pip install -r requirements.txt

Once the necessary libraries have been installed, the file createModel.py is
used to read the input data and train the models. To run this file from the
command line run:

$ python createModel.py

After the models have been created, they will be placed in the models/ folder.
The file musicgen.py is used to generate new MIDI and audio files based on the
models. The newly generated MIDI files will be placed in the outputs/ folder,
and the corresponding audio files will be placed in the audio/ folder. To run
this file from the command line run:

$ python musicgen.py

This is all that is needed to run the code. In order for the live demo to work,
the demo.py file needs to be run. In order to run the flask application to get
the demo working, from the command line run:

# FLASK_APP=demo.py flask run

In order to work the demo, select a model from the model drop down, then input
the desired number of output notes in the Output Length input box, and hit generate.
This will run output notes from the model and convert them to an audio file that
can be played on the demo page. If the reset button is hit, the inputs will clear
and a new audio file can be generated.
The sound keys file used to convert the MIDI to audio was too large so canvas would
not accept my submission, so no audio can be output from the demo. But, MIDI files
can still be generated and played with any application that accepts MIDI files.

