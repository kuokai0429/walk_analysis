# Walk Analysis

We introduce a novel method for walk pose trajectory analysis..

## 三維姿態估測推論 3D Pose Extimation Inference

--out_video_sf : Start frame of input video. <br>
--out_video_dl : Output video length. <br>
--pose3d_rotation : z_rotate y_rotate x_rotate <br>

```bash
python common/pose3d/vis_longframes.py --video 1-V2-front.mp4 --out_video_sf 0 --out_video_dl 1000 --pose3d_rotation 0 0 0
```

## 動作分析 Motion Analysis



```bash
python run.py --subject1 1-V2-side --subject2 8-V2-side --mode side
( python run.py --subject1 1-V2-front --subject2 8-V2-front --mode front )
```

