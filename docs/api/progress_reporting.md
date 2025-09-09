# Progress Reporting

The OmoAI API provides a progress reporting feature for long-running asynchronous tasks. This allows you to monitor the progress of a task as it moves through the processing pipeline.

## How it Works

When you submit an asynchronous task to the `POST /pipeline` endpoint, you will receive a `task_id`. You can use this `task_id` to poll the `GET /pipeline/status/{task_id}` endpoint to get the status of your task.

The response from the status endpoint will include a `progress` field, which is a number from 0 to 100 that represents the estimated progress of the task.

## Progress Calculation

The progress percentage is calculated based on the current stage of the pipeline:

- **Pending:** 0%
- **Preprocessing:** 10%
- **ASR:** 80%
- **Postprocessing:** 90%
- **Success:** 100%

## Example

Here is an example of a response from the `GET /pipeline/status/{task_id}` endpoint showing the progress:

```json
{
  "task_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
  "status": "running",
  "progress": 80.0,
  "result": null,
  "errors": [],
  "submitted_at": "2024-09-09T09:03:00Z",
  "started_at": "2024-09-09T09:03:05Z",
  "completed_at": null
}
```

This indicates that the task is currently in the ASR stage.
