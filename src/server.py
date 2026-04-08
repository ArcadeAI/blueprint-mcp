#!/usr/bin/env python3
import sys
import base64
import uuid
import threading
from pathlib import Path
from typing import Annotated, Optional
from datetime import datetime, timedelta
from enum import Enum

from google import genai
from google.genai import types
from arcade_mcp_server import Context, MCPApp

sys.path.insert(0, str(Path(__file__).parent))

_diagram_jobs = {}
_jobs_lock = threading.Lock()
MAX_JOBS_IN_MEMORY = 3
JOB_EXPIRY_MINUTES = 10


class JobStatus(str, Enum):
    QUEUED = "queued"
    GENERATING = "generating"
    COMPLETE = "complete"
    FAILED = "failed"


def _cleanup_old_jobs():
    cutoff = datetime.now() - timedelta(minutes=JOB_EXPIRY_MINUTES)
    with _jobs_lock:
        expired = [jid for jid, job in _diagram_jobs.items() if job.get("created", datetime.now()) < cutoff]
        for jid in expired:
            del _diagram_jobs[jid]

        if len(_diagram_jobs) > MAX_JOBS_IN_MEMORY:
            completed = [(jid, job.get("completed", datetime.min)) for jid, job in _diagram_jobs.items() if job.get("status") == JobStatus.COMPLETE]
            completed.sort(key=lambda x: x[1])
            for jid, _ in completed[:len(_diagram_jobs) - MAX_JOBS_IN_MEMORY]:
                del _diagram_jobs[jid]


def _update_job(job_id, **updates):
    with _jobs_lock:
        if job_id in _diagram_jobs:
            _diagram_jobs[job_id].update(updates)


def _upload_to_files_api(api_key, file_path):
    """Upload image to Gemini Files API, return file URI."""
    client = genai.Client(api_key=api_key)
    uploaded = client.files.upload(file=str(file_path))
    return uploaded.uri


def _generate_diagram_background(job_id, api_key, contents, aspect_ratio, resolution, filename_prefix, model):
    from generator import DiagramGenerator, GenerationConfig
    from prompts import AspectRatio, ImageSize

    try:
        _update_job(job_id, status=JobStatus.GENERATING, started=datetime.now())

        generator = DiagramGenerator(api_key=api_key, model=model)
        config = GenerationConfig(
            aspect_ratio=AspectRatio(aspect_ratio),
            image_size=ImageSize(resolution)
        )

        if isinstance(contents, str):
            result = generator.client.generate(prompt=contents, config=config, filename_prefix=filename_prefix)
        else:
            result = generator.client.generate_with_contents(contents=contents, config=config, filename_prefix=filename_prefix)

        if result.success:
            with open(result.file_path, 'rb') as f:
                image_bytes = f.read()
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')

            file_uri = _upload_to_files_api(api_key, result.file_path)
            Path(result.file_path).unlink()

            _update_job(job_id,
                status=JobStatus.COMPLETE,
                completed=datetime.now(),
                result={
                    "success": True,
                    "width": result.width,
                    "height": result.height,
                    "model": result.model_used,
                    "filename": Path(result.file_path).name,
                    "base64": image_base64,
                    "file_uri": file_uri,
                }
            )
        else:
            _update_job(job_id,
                status=JobStatus.FAILED,
                completed=datetime.now(),
                result={"success": False, "error": result.error}
            )
    except Exception as e:
        _update_job(job_id,
            status=JobStatus.FAILED,
            completed=datetime.now(),
            result={"success": False, "error": str(e)}
        )


app = MCPApp(name="blueprint_mcp", version="4.0.0", log_level="INFO")


@app.tool(requires_secrets=["GOOGLE_API_KEY"])
def start_diagram_job(
    context: Context,
    description: Annotated[str, "Text prompt describing the diagram to generate, including specific components, labels, and relationships"],
    diagram_type: Annotated[Optional[str], "Type: architecture, flowchart, data_flow, sequence, infographic, generic"] = "generic",
    aspect_ratio: Annotated[Optional[str], "Ratio: 1:1, 1:4, 1:8, 2:3, 3:2, 3:4, 4:1, 4:3, 4:5, 5:4, 8:1, 9:16, 16:9, 21:9"] = "16:9",
    resolution: Annotated[Optional[str], "Resolution: 1K, 2K, 4K"] = "2K",
    model: Annotated[Optional[str], "Model: pro (best quality, default), flash (faster iteration)"] = "pro",
) -> Annotated[str, "Job ID"]:
    """Start async diagram generation on the server. Returns a job ID. Poll with check_job_status, then retrieve the image with download_diagram."""
    from prompts import DiagramType, AspectRatio, ImageSize, optimize_prompt_for_nano_banana

    try:
        api_key = context.get_secret("GOOGLE_API_KEY")
        job_id = str(uuid.uuid4())

        try:
            dtype = DiagramType(diagram_type.lower())
        except ValueError:
            dtype = DiagramType.GENERIC

        optimized_prompt = optimize_prompt_for_nano_banana(
            description, dtype, AspectRatio(aspect_ratio), ImageSize(resolution), emphasis_on_text=True
        )

        _cleanup_old_jobs()
        with _jobs_lock:
            _diagram_jobs[job_id] = {"status": JobStatus.QUEUED, "created": datetime.now()}

        threading.Thread(
            target=_generate_diagram_background,
            args=(job_id, api_key, optimized_prompt, aspect_ratio, resolution, f"diagram_{diagram_type}", model),
            daemon=True
        ).start()

        return f"Job ID: {job_id}\nWait 30 seconds, then check_job_status"
    except Exception as e:
        return f"Error: {str(e)}"


@app.tool(requires_secrets=["GOOGLE_API_KEY"])
def edit_diagram(
    context: Context,
    file_uri: Annotated[str, "file_uri returned by a previous download_diagram call"],
    instructions: Annotated[str, "What to change: e.g. 'make the database box larger' or 'use dark mode colors'"],
    aspect_ratio: Annotated[Optional[str], "Ratio: 1:1, 1:4, 1:8, 2:3, 3:2, 3:4, 4:1, 4:3, 4:5, 5:4, 8:1, 9:16, 16:9, 21:9"] = "16:9",
    resolution: Annotated[Optional[str], "Resolution: 1K, 2K, 4K"] = "2K",
    model: Annotated[Optional[str], "Model: pro (best quality, default), flash (faster iteration)"] = "pro",
) -> Annotated[str, "Job ID to poll with check_job_status"]:
    """Edit a previously generated diagram. Pass the file_uri from download_diagram. The file_uri references the image stored in Google's infrastructure so it survives server restarts. Starts async generation. Poll with check_job_status, then download_diagram."""
    try:
        api_key = context.get_secret("GOOGLE_API_KEY")
        job_id = str(uuid.uuid4())

        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part(file_data=types.FileData(file_uri=file_uri, mime_type="image/png")),
                    types.Part(text=instructions),
                ]
            )
        ]

        _cleanup_old_jobs()
        with _jobs_lock:
            _diagram_jobs[job_id] = {"status": JobStatus.QUEUED, "created": datetime.now()}

        threading.Thread(
            target=_generate_diagram_background,
            args=(job_id, api_key, contents, aspect_ratio, resolution, "edit", model),
            daemon=True
        ).start()

        return f"Job ID: {job_id}\nWait 30 seconds, then check_job_status"
    except Exception as e:
        return f"Error: {str(e)}"


@app.tool
def check_job_status(
    context: Context,
    job_id: Annotated[str, "Job ID returned by start_diagram_job or edit_diagram"],
) -> Annotated[str, "Job status"]:
    """Check diagram generation progress. Returns one of: 'Complete' (ready to download), 'Generating' (in progress), 'Queued', or 'Failed'."""
    _cleanup_old_jobs()

    with _jobs_lock:
        if job_id not in _diagram_jobs:
            return "Job not found"
        job = dict(_diagram_jobs[job_id])

    status = job["status"]
    elapsed = (datetime.now() - job["created"]).total_seconds()

    if status == JobStatus.COMPLETE:
        return f"Complete ({elapsed:.0f}s) - Ready to download"
    elif status == JobStatus.FAILED:
        return f"Failed: {job.get('result', {}).get('error', 'Unknown')}"
    elif status == JobStatus.GENERATING:
        return f"Generating ({elapsed:.0f}s elapsed, typically 30-60s)"
    else:
        return f"Queued ({elapsed:.0f}s elapsed)"


@app.tool
def download_diagram(
    context: Context,
    job_id: Annotated[str, "Job ID returned by start_diagram_job or edit_diagram"],
) -> Annotated[str, "Pipe-delimited string: IMAGE|<filename>|<width>|<height>|<file_uri>|<base64_png_data>"]:
    """Download a completed diagram as a base64-encoded PNG image. Returns a pipe-delimited string: IMAGE|<filename.png>|<width_px>|<height_px>|<file_uri>|<base64_encoded_png>. To save the image: split by '|', base64-decode the 6th field, write to a .png file. Keep the file_uri (5th field) to make edits later with edit_diagram."""
    _cleanup_old_jobs()

    with _jobs_lock:
        if job_id not in _diagram_jobs:
            return "Job not found"
        job = dict(_diagram_jobs[job_id])

    if job["status"] != JobStatus.COMPLETE:
        return f"Job not ready (status: {job['status']})"

    result = job.get("result", {})
    if not result.get("success"):
        return f"Failed: {result.get('error')}"

    response = f"IMAGE|{result['filename']}|{result['width']}|{result['height']}|{result['file_uri']}|{result['base64']}"

    with _jobs_lock:
        if job_id in _diagram_jobs:
            del _diagram_jobs[job_id]

    return response


def main():
    transport = sys.argv[1] if len(sys.argv) > 1 else "stdio"
    app.run(transport=transport, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
