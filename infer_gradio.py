"""
RIFT-SVC 推理语训练界面

"""


import os
import re
import csv
from pathlib import Path

import logging
logging.disable(logging.CRITICAL)

import click
import gradio as gr
import gdown

# import numpy as np
import torch
import torchaudio
# import tempfile
import gc
import traceback
from slicer import Slicer

from infer import (
    load_models, 
    load_audio, 
    apply_fade, 
    batch_process_segments
)

global svc_model, vocoder, rmvpe, hubert, rms_extractor, spk2idx, dataset_cfg, device
svc_model = vocoder = rmvpe = hubert = rms_extractor = spk2idx = dataset_cfg = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.chdir(os.path.dirname(__file__))

def load_models_from_csv():
    csv_path = "./models.csv"
    models_dict = {}
    if not os.path.exists(csv_path):
        return models_dict
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        # 音色名称  模型名称 模型链接
        for row in reader:
            timbre_name = row['音色名称'].strip()
            model_name = row['模型名称'].strip()
            model_url = row['模型链接'].strip()
            if timbre_name == "" or model_name == "" or model_url == "":
                continue

            model_path = "./models/" + timbre_name + "/" + model_name
            model_dict = {}
            model_dict["model_path"] = model_path.replace("\\", "/")
            model_dict["model_url"] = model_url

            models_dict[timbre_name] = model_dict

    return models_dict

def load_models_list():
    models_root_path = "./models"
    models_dict = {}

    # 优先使用csv文件加载模型，放置第一次用了csv，第二次用的时候models已经存在了，就无法加载csv里的模型了
    csv_path = "./models.csv"
    if os.path.exists(csv_path):
        # 如果models文件夹不存在，说明不是在本地运行，那就到云端下载一份模型的列表，然后生成字典返回，等模型使用的时候再下载TODO
        models = load_models_from_csv()
        models_dict.update(models)
        return models_dict
    
    if not os.path.exists(models_root_path):
        return models_dict
        
    for folder in os.listdir(models_root_path):
        folder_path = models_root_path + "/" + folder
        if os.path.isdir(folder_path):
            timbre = folder
            
            ckpt_list = os.listdir(folder_path)
            for ckpt in ckpt_list:
                if ckpt.lower().endswith(".ckpt"):
                    model_path = folder_path + "/" + ckpt

                    model_dict = {}
                    model_dict["model_path"] = model_path.replace("\\", "/")
                    model_dict["model_url"] = ""

                    models_dict[timbre] = model_dict
                    break
            
    return models_dict
     
all_models_dict = load_models_list()

def get_drive_id(url):
    """ 通过网盘文件url获取id """
    pattern = r"(?:https?://)?(?:www\.)?drive\.google\.com/(?:file/d/|folder/d/|open\?id=|uc\?id=|drive/folders/)([a-zA-Z0-9_-]+)"
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    else:
        return url

def initialize_models(model_path):
    global svc_model, vocoder, rmvpe, hubert, rms_extractor, spk2idx, dataset_cfg
    
    # lang = LANGUAGES[current_language]
    use_fp16 = True
    
    if svc_model is not None:
        del svc_model
        del vocoder
        del rmvpe
        del hubert
        del rms_extractor
        torch.cuda.empty_cache()
        gc.collect()
    
    try:
        if not os.path.exists(model_path):
            # return [], f"{lang['error_model_not_found']}: {model_path}"
            return
        
        svc_model, vocoder, rmvpe, hubert, rms_extractor, spk2idx, dataset_cfg = load_models(model_path, device, use_fp16)
        available_speakers = list(spk2idx.keys())
        # return available_speakers, f"{lang['model_loaded_success']} {', '.join(available_speakers)}"
        return
    except Exception as e:
        error_trace = traceback.format_exc()
        # return [], f"{lang['error_loading_model']}: {str(e)}\n\n{lang['error_details_label']}: {error_trace}"
        return
        
def process_with_progress(
    progress=gr.Progress(),
    input_audio=None,
    output_file=None,
    speaker=None,
    key_shift=0,
    infer_steps=32,
    robust_f0=1,
    use_fp16=True,
    batch_size=1,
    ds_cfg_strength=0.1,
    spk_cfg_strength=1.0,
    skip_cfg_strength=0.0,
    cfg_skip_layers=6,
    cfg_rescale=0.7,
    cvec_downsample_rate=2,
    slicer_threshold=-30.0,
    slicer_min_length=3000,
    slicer_min_interval=100,
    slicer_hop_size=10,
    slicer_max_sil_kept=200
):
    global svc_model, vocoder, rmvpe, hubert, rms_extractor, spk2idx, dataset_cfg
    
    # lang = LANGUAGES[current_language]
    
    target_loudness = -18.0
    restore_loudness = True
    fade_duration = 20.0
    
    if input_audio is None:
        return None, "error_no_audio"
    
    if svc_model is None:
        return None, "error_no_model"
    
    # if speaker is None or speaker not in spk2idx:
        # return None, "error_invalid_speaker".format(", ".join(spk2idx.keys()))
    
    try:
        progress(0, desc="loading_audio")
        
        if speaker:
            speaker_id = spk2idx[speaker]
        else:
            speaker_id = list(spk2idx.values())[0]
        
        hop_length = 512
        sample_rate = 44100
        
        if cfg_skip_layers < 0:
            cfg_skip_layers_value = None
        else:
            cfg_skip_layers_value = cfg_skip_layers
        
        audio = load_audio(input_audio, sample_rate)
        
        slicer = Slicer(
            sr=sample_rate,
            threshold=slicer_threshold,
            min_length=slicer_min_length,
            min_interval=slicer_min_interval,
            hop_size=slicer_hop_size,
            max_sil_kept=slicer_max_sil_kept
        )
        
        progress(0.1, desc="slicing_audio")
        segments_with_pos = slicer.slice(audio)
        
        if not segments_with_pos:
            return None, "error_no_segments"
        
        fade_samples = int(fade_duration * sample_rate / 1000)
        
        progress(0.2, desc="start_conversion")
        
        with torch.no_grad():
            processed_segments = batch_process_segments(
                segments_with_pos, svc_model, vocoder, rmvpe, hubert, rms_extractor,
                speaker_id, sample_rate, hop_length, device,
                key_shift, infer_steps, ds_cfg_strength, spk_cfg_strength,
                skip_cfg_strength, cfg_skip_layers_value, cfg_rescale,
                cvec_downsample_rate, target_loudness, restore_loudness,
                robust_f0, use_fp16, batch_size, progress, "processing_segment"
            )
            
            result_audio = np.zeros(len(audio) + fade_samples)
            
            for idx, (start_sample, audio_out, expected_length) in enumerate(processed_segments):
                segment_progress = 0.8 + (0.1 * (idx / len(processed_segments)))
                progress(segment_progress, desc="finalizing_audio")
                
                if len(audio_out) > expected_length:
                    audio_out = audio_out[:expected_length]
                elif len(audio_out) < expected_length:
                    audio_out = np.pad(audio_out, (0, expected_length - len(audio_out)), 'constant')
                
                if idx > 0:
                    audio_out = apply_fade(audio_out.copy(), fade_samples, fade_in=True)
                    result_audio[start_sample:start_sample + fade_samples] *= np.linspace(1, 0, fade_samples)
                
                if idx < len(processed_segments) - 1:
                    audio_out[-fade_samples:] *= np.linspace(1, 0, fade_samples)
                
                result_audio[start_sample:start_sample + len(audio_out)] += audio_out
        
        progress(0.9, desc="finalizing_audio")
        result_audio = result_audio[:len(audio)]
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            output_path = temp_file.name
        
        # torchaudio.save(output_path, torch.from_numpy(result_audio).unsqueeze(0).float(), sample_rate)
        torchaudio.save(output_audio, torch.from_numpy(result_audio).unsqueeze(0).float(), \
                        sample_rate, encoding="PCM_S", bits_per_sample=24)
        
        progress(1.0, desc="processing_complete")
        return (sample_rate, result_audio), "conversion_complete".format(speaker, key_shift)
        
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            torch.cuda.empty_cache()
            gc.collect()
            
            return None, lang["error_out_of_memory"]
        else:
            return None, lang["error_conversion"].format(str(e))
    except Exception as e:
        error_trace = traceback.format_exc()
        return None, lang["error_details"].format(str(e), error_trace)
    finally:
        torch.cuda.empty_cache()
        gc.collect()


def infer_audio(sid0, key_shift, cfg_strength, inputs):
                    
    global all_models_dict
    
    if sid0 is None:
        yield "请指定推理音色", None
        return
        
    if not inputs:
        yield "请上传需转换音频", None
        return
    
    model_dict = all_models_dict[sid0]
    model_path = model_dict["model_path"]
    model_url = model_dict["model_url"]
    
    # 如果模型不存在，那可能就是云端运行，需要下载模型
    if not os.path.exists(model_path):
        # 在这里下载模型
        if model_url != "":
            file_id = get_drive_id(model_url)
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            gdown.download(id=file_id, output=model_path, fuzzy=True)
        else:
            yield "该语种的模型不存在", None
            return
    
    initialize_models(model_path)
    
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
    
    output_file_list = []
    for audio in inputs:
        output_file = output_dir + "/" + sid0 + "_" + Path(audio).stem + ".wav"
        output_file_list.append(output_file)
        process_with_progress(
            input_audio=audio,
            output_file=output_file,
            speaker=None,
            key_shift=key_shift,
            infer_steps=64,
            ds_cfg_strength=0.1,
            spk_cfg_strength=cfg_strength,
            skip_cfg_strength=0.0,
        )
        
    yield "转换完成", output_file_list
    
    return
    
with gr.Blocks(title="RIFT-SVC WebUI") as app:
    gr.Markdown("## RIFT-SVC WebUI v1.0")
    gr.Markdown(
        value = "待定"
    )
    
    def refresh_timbre():
        global all_models_dict
        
        all_models_dict.clear()
        all_models_dict = load_models_list()
        sid0s = sorted(list(all_models_dict.keys()))
        def_value = ""
        if len(sid0s) > 0:
            def_value = sid0s[0]
        return gr.update(choices=sid0s, value=def_value)
    
    with gr.Row(equal_height=True):
        with gr.Column():
            with gr.Row():
                sid0 = gr.Dropdown(label="推理音色", choices=sorted(list(all_models_dict.keys())))

                refresh_button = gr.Button("刷新音色列表", variant="primary")
                refresh_button.click(
                                fn=refresh_timbre,
                                inputs=[],
                                outputs=[sid0],
                            )
            with gr.Row():
                vc_transform1 = gr.Number(
                    label=("变调(整数, 半音数量, 升八度12降八度-12)"),
                    value=0,
                )

                cfg_strength = gr.Slider(
                    minimum=1,
                    maximum=5,
                    label=("检索特征占比"),
                    value=2,
                    step=1,
                    interactive=True,
                )

        # with gr.Column():
            # gr_audio_sample = gr.Audio(autoplay=True, sources=['upload'])
            # gr_bt_sample = gr.Button(value='试听', variant='primary')
            # gr_bt_sample.click(fn=play_sample, inputs=[gr_dd_voice_type, sid0],
                               # outputs=[gr_audio_sample])
        
    with gr.Row(equal_height=True):
        inputs = gr.File(
            file_count="multiple", label="选择待转换音频"
        )
        outputs = gr.File(
            file_count="multiple",
            file_types=['audio'],
            label=("生成音频"),
        )

    with gr.Row(equal_height=True):
        but1 = gr.Button(("转换"), variant="primary")
        download_all = gr.Button("下载所有生成音频", variant="primary")
        vc_output3 = gr.Textbox(label=("输出信息"))

        but1.click(
            infer_audio,
            [
                sid0,
                vc_transform1,
                cfg_strength,
                inputs
            ],
            [vc_output3, outputs],
            concurrency_limit=4
        )

    download_all.click(None, [], [], js="""
                        () => {
                            const component = Array.from(document.getElementsByTagName('label')).find(el => el.textContent.trim() === '生成音频').parentElement;
                            const links = component.getElementsByTagName('a');
                            for (let link of links) {
                                if (link.href.startsWith("http:") && !link.href.includes("127.0.0.1")) {
                                    link.href = link.href.replace("http:", "https:");
                                }
                                link.click();
                            }
                        }
                    """)

   
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
def main(port, host, share):
    global app
    print("Starting app...")
    app.queue().launch(server_name=host, server_port=port, share=share, debug=False, inbrowser=True)

if __name__ == "__main__":
    main()


