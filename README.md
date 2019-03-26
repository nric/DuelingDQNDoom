# DuelingDQNDoom
This is an implementation of a Dueling Deep Q Learning agent that learns to play Doom.
It is Based on VizDoom and Tenorflow.Keras. It should be compatible with TF2 but was tested
with TF-GPU 1.12 on a Ubuntu 18.10. Don't try on Windows, too many of the required packeges 
are problematic to run there.
Credit goes to : https://github.com/flyyufelix/VizDoom-Keras-RL - I just re-typed it and adapted
it to TF2 nomenclature and python 3.7 for my own learning. 
I highly recommed to use the specific vizdoom version. I had to change quite a bit from the original
flyyufelix code to get it running again because the vizdoom project is quite active.

In addtion to the the usual (tensorflow, python >=3.6) you will need to run the following:

sudo apt-get install build-essential zlib1g-dev libsdl2-dev libjpeg-dev \
nasm tar libbz2-dev libgtk2.0-dev cmake git libfluidsynth-dev libgme-dev \
libopenal-dev timidity libwildmidi-dev unzip
sudo apt-get install libboost-all-dev
pip install vizdoom==1.1.7
pip install -U scikit-image
