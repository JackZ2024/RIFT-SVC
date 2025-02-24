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

import infer_api

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
        infer_api.infer(
            model_path,
            audio,
            output_file,
            speaker = "",
            key_shift = key_shift,
            device = None,
            infer_steps=64,
            cfg_strength=cfg_strength,
            target_loudness=-18.0,
            restore_loudness=True,
            interpolate_src=0.0,
            fade_duration=20.0
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


