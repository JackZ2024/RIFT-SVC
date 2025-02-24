#!/usr/bin/env python3
"""
generate_f0.py

This script reads a meta-information JSON file containing speakers and their corresponding audio files,
extracts the fundamental frequency (f0) for each audio file using a pre-trained RMVPE model,
applies post-processing to the extracted f0, and saves the f0 as a .f0.pt file in the same directory
as the original audio file.

Usage:
    python generate_f0.py --meta-info META_INFO_JSON --data-dir DATA_DIR --model-path MODEL_PATH [OPTIONS]

Options:
    --data-dir DIRECTORY         Path to the root of the preprocessed dataset directory. (Required)
    --model-path FILE_PATH       Path to the pre-trained RMVPE model file. (Required)
    --hop-length INTEGER         Hop length for f0 extraction. (Default: 256)
    --sample-rate INTEGER        Target sample rate in Hz. (Default: 22050)
    --num-workers-per-device INTEGER  Number of workers per device for multiprocessing. (Default: 2)
    --verbose                    Enable verbose output.
"""

import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import torch
import torchaudio
import click
from tqdm import tqdm
import numpy as np
from multiprocessing import Process, Queue, current_process, cpu_count

from rift_svc.rmvpe.inference import RMVPE
from rift_svc.utils import post_process_f0


def worker_process(audio_subset, data_dir, model_path, hop_length, sample_rate, queue, verbose, device_id=0):
    """
    Worker function to extract f0 from a subset of audio files.

    Args:
        audio_subset (list): List of audio entries to process.
        data_dir (Path): Root directory of the preprocessed dataset.
        model_path (str): Path to the pre-trained RMVPE model.
        hop_length (int): Hop length for f0 extraction.
        sample_rate (int): Sample rate of the audio files in Hz.
        queue (Queue): Multiprocessing queue to communicate progress.
        verbose (bool): If True, enable verbose output.
        device_id (int): CUDA device ID.
    """
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    try:
        # Initialize RMVPE model
        rmvpe = RMVPE(model_path=model_path, hop_length=160, device=device)
    except Exception as e:
        queue.put(f"Error initializing RMVPE model in process {current_process().name}: {e}")
        return

    for audio in audio_subset:
        speaker = audio.get('speaker')
        file_name = audio.get('file_name')

        if not speaker or not file_name:
            if verbose:
                queue.put(f"Skipping invalid entry: {audio} in process {current_process().name}")
            continue

        # Construct paths
        wav_path = Path(data_dir) / speaker / f"{file_name}.wav"
        f0_path = Path(data_dir) / speaker / f"{file_name}.f0.pt"

        if not wav_path.is_file():
            if verbose:
                queue.put(f"Warning: WAV file not found: {wav_path} in process {current_process().name}")
            continue

        try:
            # Load audio
            waveform, sr = torchaudio.load(str(wav_path))  # Convert Path to string
            waveform = waveform.to(device)

            # Ensure waveform has proper shape for RMVPE (batch, samples)
            if len(waveform.shape) == 1:
                waveform = waveform.unsqueeze(0)  # Shape: (1, samples)
            elif len(waveform.shape) == 2 and waveform.shape[0] != 1:
                # Convert to mono by averaging channels
                waveform = waveform.mean(dim=0, keepdim=True)  # Shape: (1, samples)

            # Resample if necessary (assuming preprocessing handled sample rate)
            if sr != sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate).to(device)
                waveform = resampler(waveform)

            # Extract f0 using RMVPE
            f0 = rmvpe.infer_from_audio(waveform, sample_rate=sample_rate, device=device, thred=0.03, use_viterbi=False)

            # Compute number of frames based on audio length and hop_length
            n_frames = int(waveform.shape[-1] // hop_length) + 1

            # Post-process f0 to align with Mel spectrogram frames
            f0_processed = post_process_f0(
                f0=f0,
                sample_rate=sample_rate,
                hop_length=hop_length,
                n_frames=n_frames,
                silence_front=0.0  # Adjust if you have leading silence
            )

            # Save the f0 tensor
            torch.save(torch.from_numpy(f0_processed).float().cpu(), f0_path)

            if verbose:
                queue.put(f"Saved f0: {f0_path} in process {current_process().name}")
            
            # Send progress update
            queue.put("PROGRESS")

        except Exception as e:
            queue.put(f"Error processing {wav_path} in process {current_process().name}: {e}")
            continue

    queue.put(f"Process {current_process().name} completed.")

@click.command()
@click.option(
    '--data-dir',
    type=click.Path(exists=True, file_okay=False, readable=True),
    required=True,
    help='Path to the root of the preprocessed dataset directory.'
)
@click.option(
    '--model-path',
    type=click.Path(exists=True, file_okay=True, readable=True),
    default='pretrained/rmvpe/model.pt',
    show_default=True,
    help='Path to the pre-trained RMVPE model file.'
)
@click.option(
    '--hop-length',
    type=int,
    default=512,
    show_default=True,
    help='Hop length for f0 extraction.'
)
@click.option(
    '--sample-rate',
    type=int,
    default=44100,
    show_default=True,
    help='Sample rate of the audio files in Hz.'
)
@click.option(
    '--num-workers-per-device',
    type=int,
    default=1,
    show_default=True,
    help='Number of workers per device for multiprocessing.'
)
@click.option(
    '--verbose',
    is_flag=True,
    default=False,
    help='Enable verbose output.'
)
def generate_f0(data_dir, model_path, hop_length, sample_rate, num_workers_per_device, verbose):
    process(data_dir, model_path, hop_length, sample_rate, num_workers_per_device, verbose)

def process(data_dir,model_path='pretrained/rmvpe/model.pt', hop_length=512, sample_rate=44100, 
            num_workers_per_device=1, verbose=False):
    """
    Generate f0 for each audio file specified in the meta_info.json and save them as .f0.pt files.
    """
    meta_info = Path(data_dir) / "meta_info.json"
    try:
        with open(meta_info, 'r', encoding='utf-8') as f:
            meta = json.load(f)
    except Exception as e:
        click.echo(f"Error reading meta_info.json: {e}", err=True)
        sys.exit(1)

    speakers = meta.get('speakers', [])
    train_audios = meta.get('train_audios', [])
    test_audios = meta.get('test_audios', [])

    # Combine train and test audios
    all_audios = train_audios + test_audios

    if not all_audios:
        click.echo("No audio files found in meta_info.json.", err=True)
        sys.exit(1)

    # Detect number of available CUDA devices
    num_devices = torch.cuda.device_count()
    if num_devices > 1:
        devices = list(range(num_devices))
    elif num_devices == 1:
        devices = [0]
    else:
        devices = []
    
    if verbose:
        click.echo(f"Number of CUDA devices available: {num_devices}")

    if devices:
        click.echo(f"Using CUDA devices: {devices}")
    else:
        click.echo("No CUDA devices available. Using CPU.")

    # Determine total number of workers
    if devices:
        total_workers = num_devices * num_workers_per_device
        workers_per_device = num_workers_per_device
    else:
        total_workers = num_workers_per_device
        workers_per_device = 1  # CPU workers

    # Adjust number of workers if it exceeds CPU count
    if total_workers > cpu_count():
        click.echo(f"Adjusting total workers from {total_workers} to {cpu_count()} due to CPU count limitations.")
        total_workers = cpu_count()
        workers_per_device = max(1, cpu_count() // num_devices) if devices else cpu_count()

    if verbose:
        click.echo(f"Total workers: {total_workers} (Workers per device: {workers_per_device})")

    # Split audios among workers
    split_audios = [[] for _ in range(total_workers)]
    for i, audio in enumerate(all_audios):
        split_audios[i % total_workers].append(audio)

    # Create a multiprocessing Queue for communication
    queue = Queue()

    # Create and start processes
    processes = []
    for i in range(total_workers):
        if devices:
            device = devices[i % num_devices]
        else:
            device = None
        p = Process(
            target=worker_process,
            args=(
                split_audios[i],
                Path(data_dir),
                model_path,
                hop_length,
                sample_rate,
                queue,
                verbose,
                device
            ),
            name=f"Process-{i}"
        )
        p.start()
        processes.append(p)

    # Initialize tqdm progress bar
    with tqdm(total=len(all_audios), desc="Extracting f0", unit="file") as pbar:
        completed_processes = 0
        while completed_processes < total_workers:
            message = queue.get()
            if message == "PROGRESS":
                pbar.update(1)
            elif message.startswith("Saved f0") and verbose:
                pbar.set_postfix({"Last Saved": message})
            elif message.startswith("Warning") and verbose:
                pbar.write(message)
            elif message.startswith("Error"):
                pbar.write(message)
            elif message.startswith("Process") and "completed" in message:
                completed_processes += 1
                if verbose:
                    pbar.write(message)
            else:
                # Handle other messages if necessary
                if verbose:
                    pbar.write(message)

    # Ensure all processes have finished
    for p in processes:
        p.join()

    click.echo("f0 extraction complete.")


if __name__ == "__main__":
    generate_f0()