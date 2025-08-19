# Detailed and Aligned Plan for Creating a Minimal API with Litestar

This document provides a detailed, step-by-step plan to create a minimal API for the OMOAI audio transcription and summarization pipeline using the Litestar framework. This plan is aligned with the current state of the program and is designed to be very clear and provide a lot of handholding.

## 1. Project Setup and Virtual Environment

Before we start writing the API code, we need to ensure our project is set up correctly and we are using a virtual environment to manage our dependencies. We will use `uv` for this, which is a fast and modern Python package manager.

1.  **Verify Virtual Environment**: Your project is already configured to use a virtual environment with `uv`. Make sure it is activated by running the command `source .venv/bin/activate` in your terminal. You should see the name of the virtual environment in your terminal prompt.

2.  **Verify Dependencies**: Your project's dependencies are listed in the `pyproject.toml` file. `litestar` is already included in this file. To ensure all dependencies are installed, run the command `uv pip install -r requirements.txt`.

3.  **Create the API Directory**: To keep our project organized, we will create a new directory to house our API-related code. We will create a new directory named `api` inside the `src/omoai` directory. The full path will be `src/omoai/api`.

4.  **Create the Application File**: Inside the `src/omoai/api` directory, we will create a new file named `app.py`. This file will be the main entry point for our Litestar application.

## 2. API Workflow

The API is designed to be used as a series of steps, where the output of one step is used as the input for the next. Here is the intended workflow:

1.  The client sends an audio file to the `/preprocess` endpoint.
2.  The `/preprocess` endpoint returns a path to the preprocessed audio file.
3.  The client sends the path of the preprocessed audio file to the `/asr` endpoint.
4.  The `/asr` endpoint returns the raw transcript and segments.
5.  The client sends the raw transcript and segments to the `/postprocess` endpoint.
6.  The `/postprocess` endpoint returns the final punctuated transcript and summary.

Alternatively, the client can use the `/pipeline` endpoint to run all the steps in a single call.

## 3. API Structure and Controllers

To keep our API organized and easy to maintain, we will use Litestar's `Controller` feature. A controller is a class that groups related API endpoints together. We will create a controller for each major piece of functionality in our application.

-   **`MainController`**: This controller will be responsible for the main API endpoint that runs the entire audio processing pipeline, from start to finish. It will take an audio file as input and return the final transcript and summary.

-   **`PreprocessController`**: This controller will handle the API endpoint for the audio preprocessing stage. It will take an audio file as input and convert it to the correct format for the ASR model.

-   **`ASRController`**: This controller will be responsible for the Automatic Speech Recognition (ASR) stage. It will take the path to a preprocessed audio file and return the raw transcript.

-   **`PostprocessController`**: This controller will handle the final stage of the pipeline, which involves adding punctuation to the transcript and generating a summary. It will take the raw transcript from the ASR stage as input.

Each of these controllers will be defined in its own separate file within the `src/omoai/api/` directory. For example, the `MainController` will be in a file named `src/omoai/api/main_controller.py`.

## 4. Request and Response Models with Pydantic

To ensure that the data sent to and from our API is in the correct format, we will use Pydantic models. Pydantic is a library that allows us to define data schemas using Python's type hints. Litestar has excellent integration with Pydantic, which means that it will automatically validate incoming requests and outgoing responses against our Pydantic models.

### Request Models

These models define the structure of the data that our API expects to receive in the request body.

-   **`PipelineRequest`**: This model will be used for the main pipeline endpoint. It will have a single field, `audio_file`, which will be of type `UploadFile`. This tells Litestar to expect a file upload for this endpoint.

-   **`PreprocessRequest`**: This model will be used for the preprocessing endpoint. It will also have a single field, `audio_file`, of type `UploadFile`.

-   **`ASRRequest`**: This model will be used for the ASR endpoint. It will have a single field, `preprocessed_path`, which will be a string containing the path to the preprocessed audio file.

-   **`PostprocessRequest`**: This model will be used for the post-processing endpoint. It will have a single field, `asr_output`, which will be a Python dictionary. This dictionary will contain the JSON output from the ASR stage.

### Response Models

These models define the structure of the data that our API will send back in the response.

-   **`PipelineResponse`**: This model will be used for the main pipeline endpoint. It will contain the final results of the pipeline, including the `transcript` (the full punctuated transcript), the `summary` (a dictionary containing the summary of the transcript), and the `segments` (a list of transcript segments with timestamps).

-   **`PreprocessResponse`**: This model will be used for the preprocessing endpoint. It will have an `output_path` field with the path to the preprocessed file on the server.

-   **`ASRResponse`**: This model will be used for the ASR endpoint. It will contain the `transcript_raw` (the raw, unpunctuated transcript) and the `segments` (a list of transcript segments with timestamps).

-   **`PostprocessResponse`**: This model will be used for the post-processing stage. It will contain the `transcript_punct` (the punctuated transcript) and the `summary` (a dictionary containing the summary of the transcript).

## 5. API Endpoints

Here is a detailed description of the API endpoints that we will create:

-   **`POST /pipeline`**: This endpoint will run the entire audio processing pipeline. You will send a `POST` request to this endpoint with an audio file in the request body. The API will then run the preprocessing, ASR, and post-processing stages in sequence and return a `PipelineResponse` with the final transcript and summary.

-   **`POST /preprocess`**: This endpoint will run only the audio preprocessing stage. You will send a `POST` request to this endpoint with an audio file in the request body. The API will preprocess the audio and return a `PreprocessResponse` with the path to the preprocessed file.

-   **`POST /asr`**: This endpoint will run only the ASR stage. You will send a `POST` request to this endpoint with the path to a preprocessed audio file in the request body. The API will perform speech recognition and return an `ASRResponse` with the raw transcript.

-   **`POST /postprocess`**: This endpoint will run only the post-processing stage. You will send a `POST` request to this endpoint with the JSON output from the ASR stage in the request body. The API will add punctuation to the transcript and generate a summary, and return a `PostprocessResponse` with the final results.

## 6. Error Handling

Our API will use standard HTTP status codes to let you know if a request was successful or not. For example, if a request is successful, the API will return a `200 OK` status code. If there is a problem with the request, it will return a `400 Bad Request` status code. If there is a problem on the server, it will return a `500 Internal Server Error` status code.

In addition to the status code, the response body for an error will contain a `message` field with a detailed explanation of what went wrong. For example, if you forget to include the audio file in a request, the error message might say something like "The 'audio_file' field is required."

## 7. Running the API

To run the API, you will first need to make sure that you have activated the virtual environment by running `source .venv/bin/activate`. Then, you can run the API using the `litestar` command-line tool. The command to run the API is `litestar run --app src.omoai.api.app:app`.

For development, it is very useful to run the API with the `--reload` flag, like this: `litestar run --app src.omoai.api.app:app --reload`. This will automatically restart the server whenever you make a change to the code, so you don't have to stop and start the server manually every time you make a change.

## 8. Testing the API

Once the API is running, you can test it by sending requests to it using a tool like `curl`. Here are some example `curl` commands for each endpoint:

-   **Testing the `/pipeline` endpoint**:

    `curl -X POST -F "audio_file=@/path/to/your/audio.mp3" http://127.0.0.1:8000/pipeline`

-   **Testing the `/preprocess` endpoint**:

    `curl -X POST -F "audio_file=@/path/to/your/audio.mp3" http://127.0.0.1:8000/preprocess`

-   **Testing the `/asr` endpoint**:

    `curl -X POST -H "Content-Type: application/json" -d '{"preprocessed_path": "/path/to/your/preprocessed.wav"}' http://127.0.0.1:8000/asr`

-   **Testing the `/postprocess` endpoint**:

    `curl -X POST -H "Content-Type: application/json" -d '{"asr_output": {"transcript_raw": "hello world", "segments": []}}' http://127.0.0.1:8000/postprocess`

Remember to replace `/path/to/your/audio.mp3` and `/path/to/your/preprocessed.wav` with the actual paths to your audio files.