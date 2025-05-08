class BinaryMeanIoU(tf.keras.metrics.Metric):
    def __init__(self, name="mean_iou", **kwargs):
        super().__init__(name=name, **kwargs)
        self.mean_iou = tf.keras.metrics.MeanIoU(num_classes=2)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_thresh = tf.cast(y_pred > 0.5, tf.int32)
        y_true = tf.cast(y_true, tf.int32)
        self.mean_iou.update_state(y_true, y_pred_thresh)

    def result(self):
        return self.mean_iou.result()

    def reset_states(self):
        self.mean_iou.reset_states()