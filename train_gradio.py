import threading
import queue
import re
from pathlib import Path

import os
import platform
import psutil
import signal
import subprocess
import sys
import time
from glob import glob

import click
import gradio as gr
import torch
from huggingface_hub import snapshot_download
import requests
from tqdm import tqdm  # 用于显示进度条

import logging
logging.disable(logging.CRITICAL)

training_process = None
system = platform.system()
python_executable = sys.executable or "python"

os.chdir(os.path.dirname(__file__))


path_data = os.path.dirname(__file__) + "/data"
path_project_ckpts =  os.path.dirname(__file__) + "/ckpts"


# terminal
def terminate_process_tree(pid, including_parent=True):
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        # Process already terminated
        return

    children = parent.children(recursive=True)
    for child in children:
        try:
            os.kill(child.pid, signal.SIGTERM)  # or signal.SIGKILL
        except OSError:
            pass
    if including_parent:
        try:
            os.kill(parent.pid, signal.SIGTERM)  # or signal.SIGKILL
        except OSError:
            pass

def terminate_process(pid):
    if system == "Windows":
        cmd = f"taskkill /t /f /pid {pid}"
        os.system(cmd)
    else:
        terminate_process_tree(pid)

def start_training(
    config_name="finetune",
    learning_rate=5e-5,
    batch_size_per_gpu=32,
    max_steps=100000,
    save_per_steps=2000,
    test_per_steps=2000,
    eval_cfg_strength=2.0,
    save_weights_only=True,
    resume_train=False,
    tr_checkpoint="",
):
    global training_process, stop_signal

    path_project = os.path.join(path_data, config_name)
    if not os.path.isdir(path_project):
        yield (
            f"There is not project with name {config_name}",
            gr.update(interactive=True),
            gr.update(interactive=False),
        )
        return

    meta_info = os.path.join(path_project, "meta_info.json")
    if not os.path.isfile(meta_info):
        yield f"There is no file {meta_info}", gr.update(interactive=True), gr.update(interactive=False)
        return

    # Check if a training process is already running
    if training_process is not None:
        return "Train run already!", gr.update(interactive=False), gr.update(interactive=True)

    yield "start train", gr.update(interactive=False), gr.update(interactive=False)

    # Command to run the training script with the specified arguments
    cmd = (
        f"{python_executable} train.py"
        f" --config-name finetune"
        f" model=dit-1024-16"
        f" training.run_name=finetune_v3_{config_name}"
        f" dataset.data_dir=data/{config_name}"
        f" dataset.meta_info_path=data/{config_name}/meta_info.json"
        f" +training.pretrained_path=pretrained/pretrain-v3_dit-1024-16.ckpt" 
        f" training.learning_rate={learning_rate}"
        f" training.weight_decay=0.01"
        f" training.max_steps={max_steps}"
        f" training.batch_size_per_gpu={batch_size_per_gpu}"
        f" training.save_per_steps={save_per_steps}"
        f" training.test_per_steps={test_per_steps}"
        f" training.time_schedule=lognorm"
        f" +training.freeze_adaln_and_tembed=true"
        f" training.drop_spk_prob=0.0"
        f" training.logger=tensorboard"
        f" training.save_weights_only={save_weights_only}"
        # f" training.eval_cfg_strength={eval_cfg_strength}"
    )
    
    if resume_train:
        if not os.path.exists(tr_checkpoint):
            yield f"恢复训练模型不存在", gr.update(interactive=True), gr.update(interactive=False)
            return
        
        cmd += f" +training.resume_from_checkpoint={tr_checkpoint}"

    print("run command : \n" + cmd + "\n")
    try:
        if False:
            # Start the training process
            training_process = subprocess.Popen(cmd, shell=True)

            time.sleep(5)
            yield "train start", gr.update(interactive=False), gr.update(interactive=True)

            # Wait for the training process to finish
            training_process.wait()
        else:

            def stream_output(pipe, output_queue):
                try:
                    for line in iter(pipe.readline, ""):
                        output_queue.put(line)
                except Exception as e:
                    output_queue.put(f"Error reading pipe: {str(e)}")
                finally:
                    pipe.close()

            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"

            training_process = subprocess.Popen(
                cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, env=env
            )
            yield "Training started...", gr.update(interactive=False), gr.update(interactive=True)

            stdout_queue = queue.Queue()
            stderr_queue = queue.Queue()

            stdout_thread = threading.Thread(target=stream_output, args=(training_process.stdout, stdout_queue))
            stderr_thread = threading.Thread(target=stream_output, args=(training_process.stderr, stderr_queue))
            stdout_thread.daemon = True
            stderr_thread.daemon = True
            stdout_thread.start()
            stderr_thread.start()
            stop_signal = False
            while True:
                if stop_signal:
                    training_process.terminate()
                    time.sleep(0.5)
                    if training_process.poll() is None:
                        training_process.kill()
                    yield "Training stopped by user.", gr.update(interactive=True), gr.update(interactive=False)
                    break

                process_status = training_process.poll()
                # Handle stdout
                try:
                    while True:
                        output = stdout_queue.get_nowait()
                        print(output, end="")
                        if output.strip():
                            yield output, gr.update(interactive=False), gr.update(interactive=True)
                except queue.Empty:
                    pass

                # Handle stderr
                try:
                    while True:
                        error_output = stderr_queue.get_nowait()
                        print(error_output, end="")
                        if error_output.strip():
                            yield f"{error_output.strip()}", gr.update(interactive=False), gr.update(interactive=True)
                except queue.Empty:
                    pass

                if process_status is not None and stdout_queue.empty() and stderr_queue.empty():
                    if process_status != 0:
                        yield (
                            f"Process crashed with exit code {process_status}!",
                            gr.update(interactive=False),
                            gr.update(interactive=True),
                        )
                    else:
                        yield "Training complete!", gr.update(interactive=False), gr.update(interactive=True)
                    break

                # Small sleep to prevent CPU thrashing
                time.sleep(0.1)

            # Clean up
            training_process.stdout.close()
            training_process.stderr.close()
            training_process.wait()

        time.sleep(1)

        if training_process is None:
            text_info = "train stop"
        else:
            text_info = "train complete !"

    except Exception as e:  # Catch all exceptions
        # Ensure that we reset the training process variable in case of an error
        text_info = f"An error occurred: {str(e)}"

    training_process = None

    yield text_info, gr.update(interactive=True), gr.update(interactive=False)

def stop_training():
    global training_process, stop_signal

    if training_process is None:
        return "Train not run !", gr.update(interactive=True), gr.update(interactive=False)
    terminate_process_tree(training_process.pid)
    # training_process = None
    stop_signal = True
    return "train stop", gr.update(interactive=True), gr.update(interactive=False)

def get_list_projects():
    project_list = []
    for folder in os.listdir(path_data):
        path_folder = os.path.join(path_data, folder)
        if not os.path.isdir(path_folder):
            continue
        folder = folder.lower()
        project_list.append(folder)

    projects_selelect = None if not project_list else project_list[-1]

    return project_list, projects_selelect

def create_data_project(name):
    os.makedirs(os.path.join(path_data, name), exist_ok=True)
    project_list, projects_selelect = get_list_projects()
    return gr.update(choices=project_list, value=name)
    
def refresh_data_project():
    project_list, projects_selelect = get_list_projects()
    return gr.update(choices=project_list, value=projects_selelect)

def resample_normalize_data(name_project, progress=gr.Progress()):
    path_project = os.path.join(path_data, name_project)
    
    if os.path.exists(path_project):
        # resample_normalize_audios.resample_normalize_audios(path_project, target_sample_rate=44100, target_loudness=-18.0)
        return "数据重采样及归一化完成"
    else:
        return "所选工程路径不存在"

def slicer_and_resample_audio(audio_path, new_project_name, cur_project_name):
    
    if new_project_name == "":
        new_project_name = cur_project_name

    new_project_path = os.path.join(path_data, new_project_name)
    if os.path.exists(new_project_path) and len(os.listdir(new_project_path)) > 0:
        project_list, projects_selelect = get_list_projects()
        print("工程已存在，并且不为空，请修改新工程名称")
        return "工程已存在，并且不为空，请修改新工程名称", gr.update(choices=project_list, value=cur_project_name)
        
    # 创建工程文件夹    
    new_speaker_path = os.path.join(new_project_path, new_project_name)
    os.makedirs(new_speaker_path, exist_ok=True)
    
    if not os.path.exists(audio_path):
        project_list, projects_selelect = get_list_projects()
        print("音频路径不存在")
        return "音频路径不存在", gr.update(choices=project_list, value=new_project_name)
        
    # 切分音频，把切分好的音频放到speaker目录下
    print("开始切分音频")
    cmd = f"{python_executable} slicer.py {audio_path} --out {new_speaker_path} --db_thresh -40 --min_length 8000 --min_interval 500 --max_sil_kept 500"
    training_process = subprocess.Popen(cmd, shell=True)
    time.sleep(5)
    training_process.wait()
    
    project_list, projects_selelect = get_list_projects()
    print("音频切分完成")
    return "音频切分完成", gr.update(choices=project_list, value=new_project_name)

def prepare_data(name_project):
    path_project = os.path.join(path_data, name_project)
    
    # runtime\python.exe scripts\prepare_data_meta.py --data-dir %DATA_DIR%
    # runtime\python.exe scripts\prepare_mel.py --data-dir %DATA_DIR%
    # runtime\python.exe scripts\prepare_rms.py --data-dir %DATA_DIR%
    # runtime\python.exe scripts\prepare_f0.py --data-dir %DATA_DIR% --num-workers %NUM_WORKERS_PER_DEVICE%
    # runtime\python.exe scripts\prepare_cvec.py --data-dir %DATA_DIR% --num-workers %NUM_WORKERS_PER_DEVICE%
    
    # prepare_data_meta
    cmd = f"{python_executable} scripts/prepare_data_meta.py --data-dir {path_project}"
    training_process = subprocess.Popen(cmd, shell=True)
    time.sleep(5)
    yield "prepare_data_meta start"
    training_process.wait()
    
    # prepare_mel
    cmd = f"{python_executable} scripts/prepare_mel.py --data-dir {path_project}"
    training_process = subprocess.Popen(cmd, shell=True)
    time.sleep(5)
    yield "prepare_mel start"
    training_process.wait()
    
    # prepare_rms
    cmd = f"{python_executable} scripts/prepare_rms.py --data-dir {path_project}"
    training_process = subprocess.Popen(cmd, shell=True)
    time.sleep(5)
    yield "prepare_rms start"
    training_process.wait()
    
    # prepare_f0
    cmd = f"{python_executable} scripts/prepare_f0.py --data-dir {path_project} --num-workers 1"
    training_process = subprocess.Popen(cmd, shell=True)
    time.sleep(5)
    yield "prepare_f0 start"
    training_process.wait()
    
    # prepare_cvec
    cmd = f"{python_executable} scripts/prepare_cvec.py --data-dir {path_project} --num-workers 1"
    training_process = subprocess.Popen(cmd, shell=True)
    time.sleep(5)
    yield "prepare_cvec start"
    training_process.wait()
    
    yield "处理完成"
    return "处理完成"

def download_file(url, filename):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 * 1024  # 每次下载 1KB

    with open(filename, 'wb') as file, tqdm(
        total=total_size, unit='B', unit_scale=True, desc=filename
    ) as bar:
        for data in response.iter_content(block_size):
            file.write(data)
            bar.update(len(data))

    print(f"\n文件已下载到 {filename}")

def download_pre_models():
    snapshot_download(
        repo_id="Pur1zumu/RIFT-SVC-modules",
        local_dir='pretrained',
        local_dir_use_symlinks=False,  # Don't use symlinks
        local_files_only=False,        # Allow downloading new files
        ignore_patterns=["*.git*"],    # Ignore git-related files
        resume_download=True           # Resume interrupted downloads
    )
    
    url = "https://huggingface.co/Pur1zumu/RIFT-SVC-pretrained/resolve/main/pretrain-v3_dit-1024-16.ckpt"
    
    filename = "pretrained/pretrain-v3_dit-1024-16.ckpt"
    if not os.path.exists(filename):
        download_file(url, filename)
    
    return "下载完成"
    
def infer(cm_checkpoint, nfe_step, cfg_strength, input_audio):
    
    if not input_audio:
        yield None, "请上传需转换音频"
        return
        
    if not os.path.exists(cm_checkpoint):
        yield None, "所选模型不存在",
        return
    
    output_dir = "./trans_audio"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    else:
        # 把里面的东西删除
        for file in os.listdir(output_dir):
            try:
                os.remove(output_dir + "/" + file)
            except:
                pass
    
    output_file = output_dir + "/" + Path(input_audio).stem + "_rift.wav"
    
    # python infer.py \
    # --model ckpts/finetune_ckpt-v3_dit-768-12_30000steps-lr0.00005/model-step=30000.ckpt \
    # --input 0.wav \
    # --output 0_steps32_cfg0.wav \
    # --speaker speaker1 \
    # --key-shift 0 \
    # --infer-steps 32 \
    # --batch-size 4 \
    # --ds-cfg-strength 0.1 \
    # --spk-cfg-strength 0.2 \
    # --skip-cfg-strength 0.1 \
    # --cfg-skip-layers 6 \
    # --cfg-rescale 0.7 \
    # --cvec-downsample-rate 2
    
    print("开始转换")
    cmd = (
        f"{python_executable} infer.py"
        f" --model {cm_checkpoint}"
        f" --input {input_audio}"
        f" --output {output_file}"
        f" --speaker ''"
        f" --infer-steps {nfe_step}"
        f" --spk-cfg-strength {cfg_strength}"
    )
    training_process = subprocess.Popen(cmd, shell=True)
    time.sleep(5)
    training_process.wait()
    
    yield output_file, "转换完成"
    return

def get_checkpoints_project(project_name, is_gradio=True):
    if project_name is None:
        return [], ""

    if os.path.isdir(path_project_ckpts):
        
        files_checkpoints = glob(os.path.join(path_project_ckpts, f"finetune_ckpt-v2_{project_name}", "*.ckpt"))
        files_checkpoints = [f for f in files_checkpoints if "=" in os.path.basename(f)]

        # Sort regular checkpoints by number
        files_checkpoints = sorted(
            files_checkpoints, key=lambda x: int(os.path.basename(x).split("=")[1].split(".")[0])
        )
    else:
        files_checkpoints = []

    selelect_checkpoint = None if not files_checkpoints else files_checkpoints[-1]

    if is_gradio:
        return gr.update(choices=files_checkpoints, value=selelect_checkpoint)

    return files_checkpoints, selelect_checkpoint

def reduce_ckpt(re_checkpoint, ckpt, output):
    # 加载原始检查点
    if ckpt == "":
        ckpt = re_checkpoint
        
    if not os.path.exists(ckpt):
        yield f"模型路径不存在"
        return
        
    checkpoint = torch.load(ckpt, map_location="cpu")
    # print(checkpoint.keys())
    # 只保留 state_dict
    new_checkpoint = {"state_dict": checkpoint["state_dict"], "hyper_parameters": checkpoint["hyper_parameters"]}

    if output == "" or output == ckpt:
        output = ckpt.replace(".ckpt", "_reduced.ckpt").replace("=", "-")

    # 保存新的精简检查点
    torch.save(new_checkpoint, output)
    yield f"精简完成，新检查点路径： {output}"
    return

with gr.Blocks(title="RIFT-SVC WebUI") as app:
    gr.Markdown(
        """
## RIFT-SVC WebUI v1.0
"""
    )

    # with gr.Row():
        # project_name = gr.Textbox(label="工程名称", value="my_speak")
        # bt_create = gr.Button("创建新的工程")

    with gr.Row():
        projects, projects_selelect = get_list_projects()
        cm_project = gr.Dropdown(
            choices=projects, value=projects_selelect, label="当前工程", allow_custom_value=True, scale=5
        )
        refresh_project = gr.Button("刷新", scale=1)

    # bt_create.click(fn=create_data_project, inputs=[project_name], outputs=[cm_project])
    refresh_project.click(fn=refresh_data_project, inputs=[], outputs=[cm_project])

    with gr.Tabs():
        with gr.TabItem("预处理数据"):
            # gr.Markdown(
                # """```plaintext    
     # 如果数据已经处理过了，这一步可以跳过。

     # 数据格式:  
     # data/
        # │
        # ├──my_project1/
        # │   │
        # │   ├── speaker/
        # │   │   ├── audio1.wav
        # │   │   └── audio2.wav
        # │   |   ...
        # ├──my_project2/
            # │
            # ├── speaker/
            # │   ├── audio1.wav
            # │   └── audio2.wav
            # |   ...
     # ```"""
            # )
            new_project_name = gr.Textbox(label="工程名称:(填写工程名称，在切分数据时会自动创建工程文件夹，如果留空，则使用上面的当前工程)", value="")
            audio_path = gr.Text(label="音频文件或文件夹:")
            with gr.Row():
                # resample_normalize = gr.Button("重采样及归一化音频数据")
                slicer_audio = gr.Button("切分数据音频数据")
                prepare = gr.Button("处理数据")
                download_premodels = gr.Button("下载依赖模型")
            txt_info_prepare = gr.Text(label="Info", value="")

            # resample_normalize.click(
                # fn=resample_normalize_data, inputs=[cm_project], outputs=[txt_info_prepare]
            # )
            slicer_audio.click(
                fn=slicer_and_resample_audio, inputs=[audio_path, new_project_name, cm_project], outputs=[txt_info_prepare, cm_project]
            )
            prepare.click(
                fn=prepare_data, inputs=[cm_project], outputs=[txt_info_prepare]
            )
            download_premodels.click(
                fn=download_pre_models, inputs=[], outputs=[txt_info_prepare]
            )

        with gr.TabItem("微调训练"):
            # gr.Markdown("""```plaintext 
            # 待定
# ```""")

            def resume_train_change(resume_train):
                if resume_train:
                    return gr.update(visible=True), gr.update(visible=True)
                else:
                    return gr.update(visible=False), gr.update(visible=False)
                
            with gr.Row():
                batch_size_per_gpu = gr.Number(label="Batch Size per GPU", value=32)
                learning_rate = gr.Number(label="Learning Rate", value=5e-5, step=1e-5)

            with gr.Row():
                save_per_steps = gr.Number(label="Save per steps", value=2000)
                test_per_steps = gr.Number(label="Test per steps", value=2000)
                max_steps = gr.Number(label="Max Steps", value=100000)

            with gr.Row():
                # save_per_steps = gr.Number(label="Save per steps", value=2000)
                # test_per_steps = gr.Number(label="Test per steps", value=2000)
                # max_steps = gr.Number(label="Max Steps", value=100000)
                eval_cfg_strength = gr.Slider(label="Eval cfg strength", value=2.0, minimum=1.0, maximum=5.0, step=1.0, visible=False)

            with gr.Row():
                with gr.Column():
                    save_weights_only = gr.Checkbox(label="模型仅保存权重信息", value=True)
                    resume_train = gr.Checkbox(label="恢复训练", value=False)
                    
                # cd_logger = gr.Radio(label="logger", choices=["None", "wandb"], value="None")
                
                with gr.Column():
                    start_button = gr.Button("开始训练")
                    stop_button = gr.Button("停止训练", interactive=False)
                    
            txt_info_train = gr.Text(label="Info", value="")
            
            list_checkpoints, checkpoint_select = get_checkpoints_project(projects_selelect, False)
            with gr.Row():
                tr_checkpoint = gr.Dropdown(
                    choices=list_checkpoints, value=checkpoint_select, 
                    label="Checkpoints", scale=2, visible=False
                )
                tr_checkpoint_refresh = gr.Button("刷新检查点", scale=1, visible=False)
            
            # wandb_resume_id = gr.Text(label="wandb resume id", value="", visible=False)

            tr_checkpoint_refresh.click(fn=get_checkpoints_project, inputs=[cm_project], outputs=[tr_checkpoint])
            cm_project.change(fn=get_checkpoints_project, inputs=[cm_project], outputs=[tr_checkpoint])

            resume_train.change(
                fn=resume_train_change,
                inputs=[resume_train],
                outputs=[tr_checkpoint, tr_checkpoint_refresh],
            )
            
            start_button.click(
                fn=start_training,
                inputs=[
                    cm_project,
                    learning_rate,
                    batch_size_per_gpu,
                    max_steps,
                    save_per_steps,
                    test_per_steps,
                    eval_cfg_strength,
                    save_weights_only,
                    resume_train,
                    tr_checkpoint,
                ],
                outputs=[txt_info_train, start_button, stop_button],
            )
            stop_button.click(fn=stop_training, outputs=[txt_info_train, start_button, stop_button])

        with gr.TabItem("测试模型"):

            list_checkpoints, checkpoint_select = get_checkpoints_project(projects_selelect, False)
            with gr.Row():
                nfe_step = gr.Number(label="NFE Step", value=32)
                cfg_strength = gr.Slider(label="cfg Strength", value=2.0, minimum=1.0, maximum=5.0, step=1.0)

            with gr.Row():
                cm_checkpoint = gr.Dropdown(
                    choices=list_checkpoints, value=checkpoint_select, label="Checkpoints", allow_custom_value=True
                )
                with gr.Column():
                    bt_checkpoint_refresh = gr.Button("刷新检查点")
                    check_button_infer = gr.Button("推理")

            info_text = gr.Textbox(label="Info")
            orgi_audio = gr.Audio(label="原音频", type="filepath")
            gen_audio = gr.Audio(label="转换后音频", type="filepath")

            check_button_infer.click(
                fn=infer,
                inputs=[
                    cm_checkpoint,
                    nfe_step,
                    cfg_strength,
                    orgi_audio
                ],
                outputs=[gen_audio, info_text],
            )

            bt_checkpoint_refresh.click(fn=get_checkpoints_project, inputs=[cm_project], outputs=[cm_checkpoint])
            cm_project.change(fn=get_checkpoints_project, inputs=[cm_project], outputs=[cm_checkpoint])

        with gr.TabItem("精简检查点"):
            gr.Markdown("""```plaintext 
精简模型尺寸，该功能只针对模型保存的有训练数据的模型。
```""")
            list_checkpoints, checkpoint_select = get_checkpoints_project(projects_selelect, False)
            with gr.Row():
                re_checkpoint = gr.Dropdown(
                    choices=list_checkpoints, value=checkpoint_select, label="Checkpoints", allow_custom_value=True, scale=5
                )
                with gr.Column():
                    re_checkpoint_refresh = gr.Button("刷新检查点", scale=1)
                    reduce_button = gr.Button("精简", scale=1)
                
            txt_path_checkpoint = gr.Text(label="模型路径:")
            txt_path_checkpoint_small = gr.Text(label="输出模型路径:")
            txt_info_reduce = gr.Text(label="输出信息", value="")
            
            re_checkpoint_refresh.click(fn=get_checkpoints_project, inputs=[cm_project], outputs=[re_checkpoint])
            reduce_button.click(
                fn=reduce_ckpt,
                inputs=[re_checkpoint, txt_path_checkpoint, txt_path_checkpoint_small],
                outputs=[txt_info_reduce],
            )
            cm_project.change(fn=get_checkpoints_project, inputs=[cm_project], outputs=[re_checkpoint])

@click.command()
@click.option("--port", "-p", default=None, type=int, help="Port to run the app on")
@click.option("--host", "-H", default=None, help="Host to run the app on")
@click.option(
    "--share",
    "-s",
    default=False,
    is_flag=True,
    help="Share the app via Gradio share link",
)
@click.option("--api", "-a", default=True, is_flag=True, help="Allow API access")
def main(port, host, share, api):
    global app
    print("Starting app...")
    app.queue(api_open=api).launch(server_name=host, server_port=port, share=share, show_api=api, inbrowser=True)


if __name__ == "__main__":
    main()
