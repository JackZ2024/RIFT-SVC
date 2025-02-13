@echo off
setlocal enabledelayedexpansion

REM 设置默认值
set "DEFAULT_DATA_DIR=data\finetune"
set "DEFAULT_NUM_WORKERS=1"

REM 解析命令行参数
set "DATA_DIR=%~1"
if "%DATA_DIR%"=="" set "DATA_DIR=%DEFAULT_DATA_DIR%"

set "NUM_WORKERS_PER_DEVICE=%~2"
if "%NUM_WORKERS_PER_DEVICE%"=="" set "NUM_WORKERS_PER_DEVICE=%DEFAULT_NUM_WORKERS%"

REM 打印参数
echo Using DATA_DIR: %DATA_DIR%
echo Using NUM_WORKERS_PER_DEVICE: %NUM_WORKERS_PER_DEVICE%

REM 运行 Python 预处理脚本
python scripts\prepare_data_meta.py --data-dir %DATA_DIR%
python scripts\prepare_mel.py --data-dir %DATA_DIR%
python scripts\prepare_rms.py --data-dir %DATA_DIR%
python scripts\prepare_f0.py --data-dir %DATA_DIR% --num-workers-per-device %NUM_WORKERS_PER_DEVICE%
python scripts\prepare_cvec.py --data-dir %DATA_DIR% --num-workers-per-device %NUM_WORKERS_PER_DEVICE%
python scripts\prepare_whisper.py --data-dir %DATA_DIR% --num-workers-per-device %NUM_WORKERS_PER_DEVICE%

python scripts\combine_features.py --data-dir %DATA_DIR%

endlocal