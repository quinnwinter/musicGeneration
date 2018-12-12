import musicgen as mg
from keras.backend import clear_session
from flask import Flask, jsonify, request, render_template

# clear_session()
app = Flask("__name__")


# Run the demo, handle the data, predict based, convert to MIDI then audio based on demo input
def run(m, l):
    mg.handleData(mg.fileModelOutputHP[m]['filePath'], m)
    mg.outputPredict(mg.fileModelOutputHP[m]['modelName'], int(l))
    mg.toMIDI(mg.fileModelOutputHP[m]['outputName'])
    mg.convertToAudio(m, "static/demo.wav")


@app.route("/")
def display():
    return render_template("index.html")


@app.route("/predict")
def prediction():
    # Reset Global Variables
    clear_session()
    mg.model = None
    mg.allData = []
    mg.dataSequence = []
    mg.dataSequenceOut = []
    mg.midiNotes = []
    mg.uniqueNotes = None
    mg.intToNote = None
    mg.noteToInt = None
    mg.modelDataX = None
    mg.modelDataY = None
    mg.outputNotes = []

    try:
        mod = request.args["model"]
        length = request.args["length"]
    except:
        return jsonify({
            "message": "Invalid input."
        })

    print(f"Selected model: {mod} length: {length}")
    run(mod, length)

    return jsonify({
        "message": f"Generated from model {mod} with output length of {length} notes.",
        "src": "../static/demo.wav"
    })


if __name__ == "__main__":
    app.run(debug=False)
