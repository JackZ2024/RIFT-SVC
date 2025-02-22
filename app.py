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
    
with gr.Blocks() as app_infer:
    
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
        
    with gr.TabItem(("批量转换")):
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

with gr.Blocks() as app_train:
    
    # with gr.TabItem(("训练")):
    gr.Markdown(
        value=(
            "未实现"
        )
    )
    # with gr.Row():
        # exp_dir1 = gr.Textbox(label=("输入实验名"), value="mi-test")
        # sr2 = gr.Radio(
            # label=("目标采样率"),
            # choices=["40k", "48k"],
            # value="40k",
            # interactive=True,
        # )
        # if_f0_3 = gr.Radio(
            # label=("模型是否带音高指导(唱歌一定要, 语音可以不要)"),
            # choices=[True, False],
            # value=True,
            # interactive=True,
        # )
        # version19 = gr.Radio(
            # label=("版本"),
            # choices=["v1", "v2"],
            # value="v2",
            # interactive=True,
            # visible=True,
        # )
        # np7 = gr.Slider(
            # minimum=0,
            # maximum=config.n_cpu,
            # step=1,
            # label=("提取音高和处理数据使用的CPU进程数"),
            # value=int(np.ceil(config.n_cpu / 1.5)),
            # interactive=True,
        # )
    # with gr.Group():  # 暂时单人的, 后面支持最多4人的#数据处理
        # gr.Markdown(
            # value=(
                # "step2a: 自动遍历训练文件夹下所有可解码成音频的文件并进行切片归一化, 在实验目录下生成2个wav文件夹; 暂时只支持单人训练. "
            # )
        # )
        # with gr.Row():
            # trainset_dir4 = gr.Textbox(
                # label=("输入训练文件夹路径"), value="E:\\语音音频+标注\\米津玄师\\src"
            # )
            # spk_id5 = gr.Slider(
                # minimum=0,
                # maximum=4,
                # step=1,
                # label=("请指定说话人id"),
                # value=0,
                # interactive=True,
            # )
            # but1 = gr.Button(("处理数据"), variant="primary")
            # info1 = gr.Textbox(label=("输出信息"), value="")
            # but1.click(
                # preprocess_dataset,
                # [trainset_dir4, exp_dir1, sr2, np7],
                # [info1],
                # api_name="train_preprocess",
            # )
    # with gr.Group():
        # gr.Markdown(value=("step2b: 使用CPU提取音高(如果模型带音高), 使用GPU提取特征(选择卡号)"))
        # with gr.Row():
            # with gr.Column():
                # gpus6 = gr.Textbox(
                    # label=("以-分隔输入使用的卡号, 例如   0-1-2   使用卡0和卡1和卡2"),
                    # value=gpus,
                    # interactive=True,
                    # visible=F0GPUVisible,
                # )
                # gpu_info9 = gr.Textbox(
                    # label=("显卡信息"), value=gpu_info, visible=F0GPUVisible
                # )
            # with gr.Column():
                # f0method8 = gr.Radio(
                    # label=(
                        # "选择音高提取算法:输入歌声可用pm提速,高质量语音但CPU差可用dio提速,harvest质量更好但慢,rmvpe效果最好且微吃CPU/GPU"
                    # ),
                    # choices=["pm", "harvest", "dio", "rmvpe", "rmvpe_gpu"],
                    # value="rmvpe_gpu",
                    # interactive=True,
                # )
                # gpus_rmvpe = gr.Textbox(
                    # label=(
                        # "rmvpe卡号配置：以-分隔输入使用的不同进程卡号,例如0-0-1使用在卡0上跑2个进程并在卡1上跑1个进程"
                    # ),
                    # value="%s-%s" % (gpus, gpus),
                    # interactive=True,
                    # visible=F0GPUVisible,
                # )
            # but2 = gr.Button(("特征提取"), variant="primary")
            # info2 = gr.Textbox(label=("输出信息"), value="", max_lines=8)
            # f0method8.change(
                # fn=change_f0_method,
                # inputs=[f0method8],
                # outputs=[gpus_rmvpe],
            # )
            # but2.click(
                # extract_f0_feature,
                # [
                    # gpus6,
                    # np7,
                    # f0method8,
                    # if_f0_3,
                    # exp_dir1,
                    # version19,
                    # gpus_rmvpe,
                # ],
                # [info2],
                # api_name="train_extract_f0_feature",
            # )
    # with gr.Group():
        # gr.Markdown(value=("step3: 填写训练设置, 开始训练模型和索引"))
        # with gr.Row():
            # save_epoch10 = gr.Slider(
                # minimum=1,
                # maximum=50,
                # step=1,
                # label=("保存频率save_every_epoch"),
                # value=5,
                # interactive=True,
            # )
            # total_epoch11 = gr.Slider(
                # minimum=2,
                # maximum=1000,
                # step=1,
                # label=("总训练轮数total_epoch"),
                # value=20,
                # interactive=True,
            # )
            # batch_size12 = gr.Slider(
                # minimum=1,
                # maximum=40,
                # step=1,
                # label=("每张显卡的batch_size"),
                # value=default_batch_size,
                # interactive=True,
            # )
            # if_save_latest13 = gr.Radio(
                # label=("是否仅保存最新的ckpt文件以节省硬盘空间"),
                # choices=[("是"), ("否")],
                # value=("否"),
                # interactive=True,
            # )
            # if_cache_gpu17 = gr.Radio(
                # label=(
                    # "是否缓存所有训练集至显存. 10min以下小数据可缓存以加速训练, 大数据缓存会炸显存也加不了多少速"
                # ),
                # choices=[("是"), ("否")],
                # value=("否"),
                # interactive=True,
            # )
            # if_save_every_weights18 = gr.Radio(
                # label=("是否在每次保存时间点将最终小模型保存至weights文件夹"),
                # choices=[("是"), ("否")],
                # value=("否"),
                # interactive=True,
            # )
        # with gr.Row():
            # pretrained_G14 = gr.Textbox(
                # label=("加载预训练底模G路径"),
                # value="assets/pretrained_v2/f0G40k.pth",
                # interactive=True,
            # )
            # pretrained_D15 = gr.Textbox(
                # label=("加载预训练底模D路径"),
                # value="assets/pretrained_v2/f0D40k.pth",
                # interactive=True,
            # )
            # sr2.change(
                # change_sr2,
                # [sr2, if_f0_3, version19],
                # [pretrained_G14, pretrained_D15],
            # )
            # version19.change(
                # change_version19,
                # [sr2, if_f0_3, version19],
                # [pretrained_G14, pretrained_D15, sr2],
            # )
            # if_f0_3.change(
                # change_f0,
                # [if_f0_3, sr2, version19],
                # [f0method8, gpus_rmvpe,pretrained_G14, pretrained_D15],
            # )
            # gpus16 = gr.Textbox(
                # label=("以-分隔输入使用的卡号, 例如   0-1-2   使用卡0和卡1和卡2"),
                # value=gpus,
                # interactive=True,
            # )
            # but3 = gr.Button(("训练模型"), variant="primary")
            # but4 = gr.Button(("训练特征索引"), variant="primary")
            # but5 = gr.Button(("一键训练"), variant="primary")
            # info3 = gr.Textbox(label=("输出信息"), value="", max_lines=10)
            # but3.click(
                # click_train,
                # [
                    # exp_dir1,
                    # sr2,
                    # if_f0_3,
                    # spk_id5,
                    # save_epoch10,
                    # total_epoch11,
                    # batch_size12,
                    # if_save_latest13,
                    # pretrained_G14,
                    # pretrained_D15,
                    # gpus16,
                    # if_cache_gpu17,
                    # if_save_every_weights18,
                    # version19,
                # ],
                # info3,
                # api_name="train_start",
            # )
            # but4.click(train_index, [exp_dir1, version19], info3)
            # but5.click(
                # train1key,
                # [
                    # exp_dir1,
                    # sr2,
                    # if_f0_3,
                    # trainset_dir4,
                    # spk_id5,
                    # np7,
                    # f0method8,
                    # save_epoch10,
                    # total_epoch11,
                    # batch_size12,
                    # if_save_latest13,
                    # pretrained_G14,
                    # pretrained_D15,
                    # gpus16,
                    # if_cache_gpu17,
                    # if_save_every_weights18,
                    # version19,
                    # gpus_rmvpe,
                # ],
                # info3,
                # api_name="train_start_all",
            # )

with gr.Blocks() as app_reduce:
    
    gr.Markdown(
        value=(
            "未实现"
        )
    )

# with gr.TabItem(("ckpt处理")):
    # with gr.Group():
        # gr.Markdown(value=("模型融合, 可用于测试音色融合"))
        # with gr.Row():
            # ckpt_a = gr.Textbox(label=("A模型路径"), value="", interactive=True)
            # ckpt_b = gr.Textbox(label=("B模型路径"), value="", interactive=True)
            # alpha_a = gr.Slider(
                # minimum=0,
                # maximum=1,
                # label=("A模型权重"),
                # value=0.5,
                # interactive=True,
            # )
        # with gr.Row():
            # sr_ = gr.Radio(
                # label=("目标采样率"),
                # choices=["40k", "48k"],
                # value="40k",
                # interactive=True,
            # )
            # if_f0_ = gr.Radio(
                # label=("模型是否带音高指导"),
                # choices=[("是"), ("否")],
                # value=("是"),
                # interactive=True,
            # )
            # info__ = gr.Textbox(
                # label=("要置入的模型信息"), value="", max_lines=8, interactive=True
            # )
            # name_to_save0 = gr.Textbox(
                # label=("保存的模型名不带后缀"),
                # value="",
                # max_lines=1,
                # interactive=True,
            # )
            # version_2 = gr.Radio(
                # label=("模型版本型号"),
                # choices=["v1", "v2"],
                # value="v1",
                # interactive=True,
            # )
        # with gr.Row():
            # but6 = gr.Button(("融合"), variant="primary")
            # info4 = gr.Textbox(label=("输出信息"), value="", max_lines=8)
        # but6.click(
            # merge,
            # [
                # ckpt_a,
                # ckpt_b,
                # alpha_a,
                # sr_,
                # if_f0_,
                # info__,
                # name_to_save0,
                # version_2,
            # ],
            # info4,
            # api_name="ckpt_merge",
        # )  # def merge(path1,path2,alpha1,sr,f0,info):
    # with gr.Group():
        # gr.Markdown(value=("修改模型信息(仅支持weights文件夹下提取的小模型文件)"))
        # with gr.Row():
            # ckpt_path0 = gr.Textbox(
                # label=("模型路径"), value="", interactive=True
            # )
            # info_ = gr.Textbox(
                # label=("要改的模型信息"), value="", max_lines=8, interactive=True
            # )
            # name_to_save1 = gr.Textbox(
                # label=("保存的文件名, 默认空为和源文件同名"),
                # value="",
                # max_lines=8,
                # interactive=True,
            # )
        # with gr.Row():
            # but7 = gr.Button(("修改"), variant="primary")
            # info5 = gr.Textbox(label=("输出信息"), value="", max_lines=8)
        # but7.click(
            # change_info,
            # [ckpt_path0, info_, name_to_save1],
            # info5,
            # api_name="ckpt_modify",
        # )
    # with gr.Group():
        # gr.Markdown(value=("查看模型信息(仅支持weights文件夹下提取的小模型文件)"))
        # with gr.Row():
            # ckpt_path1 = gr.Textbox(
                # label=("模型路径"), value="", interactive=True
            # )
            # but8 = gr.Button(("查看"), variant="primary")
            # info6 = gr.Textbox(label=("输出信息"), value="", max_lines=8)
        # but8.click(show_info, [ckpt_path1], info6, api_name="ckpt_show")
    # with gr.Group():
        # gr.Markdown(
            # value=(
                # "模型提取(输入logs文件夹下大文件模型路径),适用于训一半不想训了模型没有自动提取保存小文件模型,或者想测试中间模型的情况"
            # )
        # )
        # with gr.Row():
            # ckpt_path2 = gr.Textbox(
                # label=("模型路径"),
                # value="E:\\codes\\py39\\logs\\mi-test_f0_48k\\G_23333.pth",
                # interactive=True,
            # )
            # save_name = gr.Textbox(
                # label=("保存名"), value="", interactive=True
            # )
            # sr__ = gr.Radio(
                # label=("目标采样率"),
                # choices=["32k", "40k", "48k"],
                # value="40k",
                # interactive=True,
            # )
            # if_f0__ = gr.Radio(
                # label=("模型是否带音高指导,1是0否"),
                # choices=["1", "0"],
                # value="1",
                # interactive=True,
            # )
            # version_1 = gr.Radio(
                # label=("模型版本型号"),
                # choices=["v1", "v2"],
                # value="v2",
                # interactive=True,
            # )
            # info___ = gr.Textbox(
                # label=("要置入的模型信息"), value="", max_lines=8, interactive=True
            # )
            # but9 = gr.Button(("提取"), variant="primary")
            # info7 = gr.Textbox(label=("输出信息"), value="", max_lines=8)
            # ckpt_path2.change(
                # change_info_, [ckpt_path2], [sr__, if_f0__, version_1]
            # )
        # but9.click(
            # extract_small_model,
            # [ckpt_path2, save_name, sr__, if_f0__, info___, version_1],
            # info7,
            # api_name="ckpt_extract",
        # )

with gr.Blocks(title="RIFT-SVC WebUI") as app:
    gr.Markdown("## RIFT-SVC WebUI v1.0")
    gr.Markdown(
        value = "待定"
    )
    
    gr.TabbedInterface(
        [app_infer, app_train, app_reduce],
        ["模型推理", "模型训练", "精简模型"],
    )
   
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
@click.option(
    "--root_path",
    "-r",
    default=None,
    type=str,
    help='The root path (or "mount point") of the application, if it\'s not served from the root ("/") of the domain. Often used when the application is behind a reverse proxy that forwards requests to the application, e.g. set "/myapp" or full URL for application served at "https://example.com/myapp".',
)
def main(port, host, share, api, root_path):
    global app
    print("Starting app...")
    app.queue(api_open=api).launch(server_name=host, server_port=port, share=share, show_api=api, root_path=root_path)

if __name__ == "__main__":
    # main()
    app.queue().launch(debug=False, inbrowser=True)


