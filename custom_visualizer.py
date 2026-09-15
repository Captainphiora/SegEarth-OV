import numpy as np
from typing import Optional

import mmcv
from mmseg.registry import VISUALIZERS
from mmseg.structures import SegDataSample
from mmseg.visualization import SegLocalVisualizer


@VISUALIZERS.register_module()
class TripletSegLocalVisualizer(SegLocalVisualizer):
    """在原版基础上，把纯原图作为最左栏拼进可视化结果。

    输出布局: [ 原图 | 原图+真值掩码 | 原图+预测掩码 ]
    其余行为与 mmseg 的 SegLocalVisualizer 完全一致。
    """

    def add_datasample(
            self,
            name: str,
            image: np.ndarray,
            data_sample: Optional[SegDataSample] = None,
            draw_gt: bool = True,
            draw_pred: bool = True,
            show: bool = False,
            wait_time: float = 0,
            out_file: Optional[str] = None,
            step: int = 0,
            with_labels: Optional[bool] = True) -> None:
        classes = self.dataset_meta.get('classes', None)
        palette = self.dataset_meta.get('palette', None)

        panels = [image]  # 最左栏: 纯原图

        if draw_gt and data_sample is not None and 'gt_sem_seg' in data_sample:
            assert classes is not None, 'class information is not provided.'
            panels.append(
                self._draw_sem_seg(image, data_sample.gt_sem_seg, classes,
                                   palette, with_labels))

        if (draw_pred and data_sample is not None
                and 'pred_sem_seg' in data_sample):
            assert classes is not None, 'class information is not provided.'
            panels.append(
                self._draw_sem_seg(image, data_sample.pred_sem_seg, classes,
                                   palette, with_labels))

        drawn_img = np.concatenate(panels, axis=1)

        if show:
            self.show(drawn_img, win_name=name, wait_time=wait_time)

        if out_file is not None:
            mmcv.imwrite(mmcv.rgb2bgr(drawn_img), out_file)
        else:
            self.add_image(name, drawn_img, step)
