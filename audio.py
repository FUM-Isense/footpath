from openal import *

def main():
    # Initialize OpenAL context
    oalInit()

    # Load the beep sound
    try:
        source = oalOpen("/home/redha/Projects/Vision_Assistance_Forest/new_beep.wav")
    except Exception as e:
        print(f"Failed to load beep.wav: {e}")
        oalQuit()
        return

    # Set the listener's position (user position)
    listener = oalGetListener()
    listener.set_position([0, 0, 0])

    # Set the source position (direction to go)
    source.set_position([1, 0, 0])  # Adjust the position as needed

    # Play the beep sound
    source.play()

    # Keep the context alive while the sound is playing
    while source.get_state() == AL_PLAYING:
        continue

    # Clean up OpenAL context
    oalQuit()

if __name__ == "__main__":
    main()
