# Preparing data for the MIT Adobe FiveK Dataset with Lightroom

## Getting the Data

- Download the dataset from [website](https://data.csail.mit.edu/graphics/fivek/). (The "single archive (~50GB, SHA1)" or "by parts", either is fine).
- Extract the data.
- Open the file `fivek.lrcat` using Lightroom.

## Generating the Input Images

- In the `Collections` list, select the collection `InputZeroed with ExpertC WhiteBalance`.
- Select all images in the bottom (select one and press `Ctrl-A`), right-click on any of them, choose `Export/Export...`.
  - In `File Settings`, set `Image Format`=`JPEG`, `Quality`=100, `Color Space`=`sRGB`. Uncheck the `Limit File Size To` option.
  - In `Image Sizing`:
    - For full-resolution dataset, uncheck the `Resize to Fit` option.
    - For 480P dataset, set `Resize to Fit`=`Short Edge`. Check `Don't Enlarge`. Fill in `480` `pixels`.
  - Finally, click `Export`. You may need to wait for a while (~60 minutes).

## Generating the Target Images (Expert C)

- In the `Collections` list, select the collection `Experts/C`.
- Select all images in the bottom (select one and press `Ctrl-A`), right-click on any of them, choose `Export/Export...`.
  - In `File Settings`, set `Image Format`=`JPEG`, `Quality`=100, `Color Space`=`sRGB`. Uncheck the `Limit File Size To` option.
  - In `Image Sizing`:
    - For full-resolution dataset, uncheck the `Resize to Fit` option.
    - For 480P dataset, set `Resize to Fit`=`Short Edge`. Check `Don't Enlarge`. Fill in `480` `pixels`.
  - Finally, click `Export`. You may need to wait for a while (~60 minutes).