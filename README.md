# EmotionRecognition


## Preprocessing

### Duplicates

1853 duplicates removed - 5.16% of the dataset

We can efficiently find and remove duplicate images with `numpy.unique`, with 35887 images, this takes only a few seconds

### Similar Images

4443 similar removed - 12.38% of the dataset

We essentially define a similar image as the same person in the same pose, though the image may be translated, stretched, blurred, have different contrast, a watermark/text, or noise.

We find these similar images using `structural_similarity` from `skimage.metrics` in a pairwise fashion using threshold 0.6. Unfortunately, each image must be compared with each other in $\mathcal{O}(n^2)$, taking over a day with 35887 images.

One way to do this more efficiently would be to first train an image reconstruction neural network, then use the encoder to create codes for each image. Whichever codes show small distances might be similar images. Perhaps a clustering algorithm such as K-means could be used to create the groups.

We then string together similar images into groups. For example, if the following images were similar `(1, 2), (1, 3), (4, 5), (4, 9)`, then the groups would be `[1, 2, 3], [4, 5, 9]`. The benefit of this is it's quick to calculate and is good at grouping all similar images of a person together.

However, there is a balancing act to the similarity threshold - too low and the massive groups of people may string together, despite being different people. Too high, and the groups will be small, not including all of a person's similar images.

At this point, we can look at the groups! Lots of celebrity images like those of Barrack Obama, Justin Bieber, Beyoncé, and Daniel Radcliff! However, we aren't done. We must now select an image from each group to keep in the dataset, and discard the others. Or, if the group consists of different people, select all which we want to keep.

This was done manually for best results. That way we could select for no watermark (or least intrusive watermark), best facial framing (zoom/translation), no stretching, good contrast, little noise. This was somewhat time consuming, selecting the indices for the images to keep, 'none' to keep none of  them, or 'all' to keep all of them. However, a random image could technically be chosen from each group, although this would sacrifice both the quality and quantity of the data.

