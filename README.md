# ml_data_augmentation

I wrote a function that can shift an image in any direction (left, right, up, or down) by one pixel.
Then, for each image in the training set, create four shifted copies (one per direction) and add them to the training set. Finally, train your best model on this expanded training set and measure its accuracy on the test set.This technique of artificially growing the training set is called data augmentation or training set expansion.
