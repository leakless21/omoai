# ASRResponse

The `ASRResponse` model is the response object for the Automatic Speech Recognition (ASR) endpoint. It contains the transcribed text and its segmentation.

## Fields

| Field            | Type   | Description                                                                                                              |
| :--------------- | :----- | :----------------------------------------------------------------------------------------------------------------------- |
| `segments`       | `list` | A list of segments, where each segment is an object containing the transcribed text for a specific portion of the audio. |
| `transcript_raw` | `str`  | The raw, unpunctuated transcription of the audio. This field is optional.                                                |
