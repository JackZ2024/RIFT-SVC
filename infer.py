
import click
import infer_api


@click.command()
@click.option('--model', type=click.Path(exists=True), required=True, help='Path to model checkpoint')
@click.option('--input', type=click.Path(exists=True), required=True, help='Input audio file')
@click.option('--output', type=click.Path(), required=True, help='Output audio file')
@click.option('--speaker', type=str, required=True, help='Target speaker')
@click.option('--key-shift', type=int, default=0, help='Pitch shift in semitones')
@click.option('--device', type=str, default=None, help='Device to use (cuda/cpu)')
@click.option('--infer-steps', type=int, default=32, help='Number of inference steps')
@click.option('--cfg-strength', type=float, default=0.0, help='Classifier-free guidance strength')
@click.option('--target-loudness', type=float, default=-18.0, help='Target loudness in LUFS for normalization')
@click.option('--restore-loudness', is_flag=True, default=False, help='Restore loudness to original')
@click.option('--interpolate-src', type=float, default=0.0, help='Interpolate source audio')
@click.option('--fade-duration', type=float, default=20.0, help='Fade duration in milliseconds')
def main(
    model,
    input,
    output,
    speaker,
    key_shift,
    device,
    infer_steps,
    cfg_strength,
    target_loudness,
    restore_loudness,
    interpolate_src,
    fade_duration
):
    infer_api.infer(
            model,
            input,
            output,
            speaker,
            key_shift = key_shift,
            device = device,
            infer_steps=infer_steps,
            cfg_strength=cfg_strength,
            target_loudness=target_loudness,
            restore_loudness=restore_loudness,
            interpolate_src=interpolate_src,
            fade_duration=fade_duration
        )


if __name__ == '__main__':
    main()