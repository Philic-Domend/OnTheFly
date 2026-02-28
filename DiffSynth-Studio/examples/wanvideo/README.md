Step1: Download wan2.2 5b TI2V from modelscope[https://www.modelscope.cn/models/Wan-AI/Wan2.2-TI2V-5B] (T5 text encoder, wan vae)

Step2: Download Wan model finetuned on libero from modelscope

```
modelscope download --model 'MaHaoxiang/wan_libero' --token <xxxxx>
```

Step3

```
python DiffSynth-Studio/examples/wanvideo/model_inference/Wan2.2-5b-libero.py
```