from music21 import midi, stream, note, chord, converter
import glob
import numpy as np
from keras.models import load_model
from midi2audio import FluidSynth


# Output length of MIDI Data
outputLength = 32

# Files paths and hyper parameters
fileModelOutputHP = {
    'EDM': {'filePath': glob.glob("MIDIData/EDM/*.mid"),
            'modelName': 'models/edmFinal.h5',
            'outputName': 'outputs/edmFinal.mid',
            'audioName': 'audio/edmFinal.wav',
            'sequenceLength': 4,
            'LSTMDimension': 128,
            'dropoutOne': 0.1,
            'dropoutTwo': 0.1,
            'num_epochs': 100,
            'batch_size': 4
            },
    'Bach': {'filePath': glob.glob("MIDIData/Bach/*.mid"),
             'modelName': 'models/bachFinal.h5',
             'outputName': 'outputs/bachFinal.mid',
             'audioName': 'audio/bachFinal.wav',
             'sequenceLength': 4,
             'LSTMDimension': 128,
             'dropoutOne': 0.1,
             'dropoutTwo': 0.1,
             'num_epochs': 100,
             'batch_size': 4
             },
    'Beethoven': {'filePath': glob.glob("MIDIData/Beethoven/*.mid"),
                  'modelName': 'models/beethovenFinal.h5',
                  'outputName': 'outputs/beethovenFinal.mid',
                  'audioName': 'audio/beethovenFinal.wav',
                  'sequenceLength': 4,
                  'LSTMDimension': 128,
                  'dropoutOne': 0.1,
                  'dropoutTwo': 0.1,
                  'num_epochs': 100,
                  'batch_size': 4
                  },
    'Mozart': {'filePath': glob.glob("MIDIData/Mozart/*.mid"),
               'modelName': 'models/mozartFinal.h5',
               'outputName': 'outputs/mozartFinal.mid',
               'audioName': 'audio/mozartFinal.wav',
               'sequenceLength': 4,
               'LSTMDimension': 128,
               'dropoutOne': 0.1,
               'dropoutTwo': 0.1,
               'num_epochs': 100,
               'batch_size': 4
               },
    'Combined': {'filePath': glob.glob("MIDIData/*/*.mid"),
                 'modelName': 'models/combinedFinal.h5',
                 'outputName': 'outputs/combinedFinal.mid',
                 'audioName': 'audio/combinedFinal.wav',
                 'sequenceLength': 4,
                 'LSTMDimension': 128,
                 'dropoutOne': 0.1,
                 'dropoutTwo': 0.1,
                 'num_epochs': 50,
                 'batch_size': 16
                 },
}


# Creates of sequence of MIDI data with sequenceLength notes as the input (features),
# and the next note as the output (targets)
def createSequence(totalNotes, index):
    global midiNotes
    for i in range(0, totalNotes - fileModelOutputHP[index]['sequenceLength']):
        inputList = midiNotes[i: i + fileModelOutputHP[index]['sequenceLength']]
        outputList = midiNotes[i + fileModelOutputHP[index]['sequenceLength']]
        inputArr = []
        # for o, n, mn, d, t in inputList:
        for o, n, mn, t in inputList:
            # inputArr.append(noteToInt[n])
            # s = str(t) + " " + str(n) + " " + str(d)
            s = str(t) + " " + str(n)
            inputArr.append(s)
        # print(inputArr)
        # print(noteToInt[outputList[1]])
        dataSequence.append(inputArr)
        # dataSequenceOut.append(noteToInt[outputList[1]])
        # t = str(outputList[4]) + " " + str(outputList[1]) + " " + str(outputList[3])
        t = str(outputList[3]) + " " + str(outputList[1])
        dataSequenceOut.append(t)


# Reads in the MIDI Data and populates the appropriate data structures
def handleData(fp, index):
    global uniqueNotes, midiNotes, modelDataX, modelDataY, noteToInt, intToNote

    # Loop through every MIDI file in the file path
    for midiFile in fp:
        midiNotes = []
        print(midiFile)
        song = converter.parse(midiFile)
        song = song.stripTies()
        # song.show("text")

        for n in song.recurse().notesAndRests:
            if n.isNote:
                # Make the offset an integer so it can be sorted,
                # offset used to make sure notes and chords lineup correctly
                # midiNotes.append([int(n.offset * 10), n.name, n, float(n.duration.quarterLength * 4), "NOTE"])
                midiNotes.append([int(n.offset * 10), n.nameWithOctave, n, "NOTE"])

            elif n.isChord:
                chordNotes = ""
                for x in n._notes:
                    chordNotes += x.nameWithOctave + " "
                chordNotes = chordNotes[:-1]
                # midiNotes.append([int(n.offset * 10), chordNotes, n, float(n.duration.quarterLength * 4), "CHORD"])
                midiNotes.append([int(n.offset * 10), chordNotes, n, "CHORD"])

            elif n.isRest:
                # midiNotes.append([int(n.offset * 10), n.name, n, float(n.duration.quarterLength * 4), "REST"])
                midiNotes.append([int(n.offset * 10), n.name, n, "REST"])

        for x in midiNotes:
            # print(str(x[0]) + " 1:" + str(x[1]) + " 2:" + str(x[2]) + " 3:" + str(x[3]))
            # s with duration included
            # s = str(x[4]) + " " + str(x[1]) + " " + str(x[3])
            # s with no duration
            s = str(x[3]) + " " + str(x[1])
            allData.append(s)

        createSequence(len(midiNotes), index)

    # Data Structures to convert notes, chords, and rests to Integers
    uniqueNotes = (set(allData))

    noteToInt = dict((n, i) for i, n in enumerate(uniqueNotes))
    intToNote = dict((i, n) for i, n in enumerate(uniqueNotes))

    # Convert sequences of notes, chords, rests to int values
    for i in range(len(dataSequence)):
        for j in range(len(dataSequence[i])):
            dataSequence[i][j] = noteToInt[str(dataSequence[i][j])]
        dataSequenceOut[i] = noteToInt[str(dataSequenceOut[i])]

    # Prepare MIDI Data
    zeroArrayX = (len(dataSequence), fileModelOutputHP[index]['sequenceLength'], len(uniqueNotes))
    modelDataX = np.zeros(zeroArrayX)

    zeroArrayY = (len(dataSequenceOut), len(uniqueNotes))
    modelDataY = np.zeros(zeroArrayY)

    # One hot encode input sequence modelDataX
    for i in range(0, len(dataSequence)):
        for j in range(0, len(dataSequence[i])):
            modelDataX[i][j][dataSequence[i][j]] = 1

    # One hot encode output sequence modelDataY
    for i in range(0, len(dataSequenceOut)):
        modelDataY[i][dataSequenceOut[i]] = 1


# Model output back to MIDI
def outputPredict(name, length):
    global uniqueNotes, intToNote, modelDataX, outputNotes
    print("OUTPUT MIDI")
    outputNotes = []

    loadedModel = load_model(name)

    # Pick a random starting point
    startIndex = np.random.randint(0, len(modelDataX) - 1)
    pattern = modelDataX[startIndex]

    # Predict length notes from the model and save them to outputNotes array
    for i in range(0, length):
        inputX = np.reshape(pattern, (1, len(pattern), len(uniqueNotes)))
        predict = loadedModel.predict(inputX)
        predictionIndex = np.argmax(predict)
        predictedNote = intToNote[predictionIndex]

        print(predictedNote)

        # Add predicted note to output Notes
        outputNotes.append(predictedNote)

        predictedOneHot = np.zeros((1, len(uniqueNotes)))
        predictedOneHot[0][predictionIndex] = 1

        # Create new One Hot array, and copy end of old pattern plus predicted
        newPattern = np.zeros((len(pattern), len(uniqueNotes)))

        for j in range(0, len(pattern) - 1):
            newPattern[j] = pattern[j + 1]

        newPattern[-1] = predictedOneHot
        pattern = newPattern


# Takes the predicted notes and converts them to a MIDI stream using music21's stream library
def toMIDI(outName):
    global outputNotes
    # Export to MIDI using music21 stream
    outputStreamNotes = stream.Stream()

    # Add the notes from the outputNotes to the music21 stream
    for n in outputNotes:
        d = n.split()
        # original length of note
        # length = float(d[-1])
        # new random length
        length = np.random.randint(1, 3)
        # random velocity
        velocity = np.random.randint(40, 50)
        if d[0] == "NOTE":
            n = note.Note(d[1], quarterLength=length)
            n.volume.velocity = velocity
            outputStreamNotes.append(n)
        elif d[0] == "CHORD":
            # outputStreamNotes.append(chord.Chord(d[1:-1], quarterLength=length))
            c = chord.Chord(d[1:], quarterLength=(length * 2))
            c.volume.velocity = velocity
            outputStreamNotes.append(c)
        elif d[0] == "REST":
            outputStreamNotes.append(note.Rest(quarterLength=length))

    # Convert stream into MIDI file and write to output MIDI file
    midiOutNotes = midi.translate.streamToMidiFile(outputStreamNotes)
    midiOutNotes.open(outName, "wb")
    midiOutNotes.write()
    midiOutNotes.close()


# Convert the MIDI output to audio using music2audio
# Sound key file is too large for canvas, so I had to delete it from
# my submission, but the MIDI to audio conversion was only used for the demo.
# def convertToAudio(index, n):
    # fs = FluidSynth('audio/Nice-Keys-B-Plus-JN1.4.sf2')
    # fs.midi_to_audio(fileModelOutputHP[index]['outputName'], n)


# Main Function
if __name__ == "__main__":
    for idx in fileModelOutputHP:
        # Reset Global Variables
        model = None
        allData = []
        dataSequence = []
        dataSequenceOut = []
        midiNotes = []
        uniqueNotes = None
        intToNote = None
        noteToInt = None
        modelDataX = None
        modelDataY = None
        outputNotes = []

        handleData(fileModelOutputHP[idx]['filePath'], idx)
        outputPredict(fileModelOutputHP[idx]['modelName'], outputLength)
        toMIDI(fileModelOutputHP[idx]['outputName'])
        # convertToAudio(idx, fileModelOutputHP[idx]['audioName'])
