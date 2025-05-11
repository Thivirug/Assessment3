import tensorflow as tf

class BCEDiceLoss(tf.keras.metrics.Metric):
    """
        Custom metric class that combines Binary Cross Entropy and Dice Loss.
    """

    def __init__(self, name='bce_dice_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.bce_dice_loss = self.add_weight(name='bce_dice_loss', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Update the state of the metric.
        
        Args:
            y_true: Ground truth segmentation mask
            y_pred: Predicted segmentation mask
            sample_weight: Optional sample weighting
        """
        # Calculate BCE loss
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)(y_true, y_pred)
        
        # Calculate Dice loss, flatten the tensors
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        
        # Calculate intersection and union
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        smooth = 1.0  # To avoid division by zero
        dice = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
        dice_loss = 1 - dice
        
        # Combine losses
        combined_loss = bce + dice_loss
        
        # Update state
        batch_size = tf.cast(tf.shape(y_true)[0], dtype=tf.float32)
        self.bce_dice_loss.assign_add(combined_loss * batch_size)
        self.count.assign_add(batch_size)
    
    def result(self):
        """
            Return the mean loss across all examples seen so far.
        """
        return self.bce_dice_loss / self.count if self.count > 0 else 0.0
    
    def reset_states(self):
        """
            Reset the state of the metric.
        """
        self.bce_dice_loss.assign(0.0)
        self.count.assign(0.0)
        
    def __call__(self, y_true, y_pred):
        """
        Calculate and return the combined BCE and Dice loss.
        This allows the class to be used as a loss function during model.compile()
        
        Args:
            y_true: Ground truth segmentation mask
            y_pred: Predicted segmentation mask
        """
        # Calculate BCE loss
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)(y_true, y_pred)
        
        # Flatten the tensors for Dice loss
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        
        # Calculate intersection and union
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        smooth = 1.0  # To avoid division by zero
        dice = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
        dice_loss = 1 - dice
        
        # Combine and return the losses
        return bce + dice_loss