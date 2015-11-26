import os.path
import argparse
import scipy.io.wavfile
from classifier import load_model, predict

def check_format(fn):
    """raise exception if the file is not WAV format"""
    if os.path.splitext(fn)[-1] != ".wav":
        raise ValueError("Input file should be in WAV format")

def main():
    parser = argparse.ArgumentParser(description="classify the instrument in the music sample")
    parser.add_argument("-i", dest="filename", required=True, help="input file in WAV format")
    args = parser.parse_args()
    filename = args.filename
    check_format(filename)

    model_instrument = load_model("svm_instrument")
    model_family = load_model("svm_family")
    scaler = load_model("scaler_instrument")
    print("Family: %s" % predict(model_family, filename, scaler))
    print("Instrument: %s" % predict(model_instrument, filename, scaler))



if __name__ == "__main__":
    main()
