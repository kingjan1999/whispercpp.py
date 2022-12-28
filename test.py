from whispercpp import Whisper


def main():
    w = Whisper('large', './tmp-models')
    w.set_params(proc_count=1)

    result = w.transcribe("test.mp3")
    text = w.extract_text(result)
    print(text)


if __name__ == "__main__":
    main()
