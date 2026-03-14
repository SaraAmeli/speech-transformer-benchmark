import whisper


class WhisperModel:

    def __init__(self, model_size="base"):

        print(f"\nLoading model: {model_size}")
        self.model = whisper.load_model(model_size)

    def transcribe(self, audio):

        result = self.model.transcribe(audio)

        return result["text"]
