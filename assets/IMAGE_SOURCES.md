# Image sources

The image release contains the fixed subset used by this captioning experiment:

- 10,000 original MS COCO images, unchanged.
- 188 DrawBench images from PQPP: two SDXL images for each of the 94 prompts
  with SDXL ground-truth score exactly 2. No GLIDE images are included.

Source project: [Eduard6421/PQPP](https://github.com/Eduard6421/PQPP).
The [upstream README](https://github.com/Eduard6421/PQPP#license) describes
the licenses and links to the original image bundles.

COCO images retain their original Flickr licenses and terms. They are not all
covered by the license for COCO annotations. See the
[COCO terms of use](https://cocodataset.org/#termsofuse) and
[Flickr Creative Commons information](https://www.flickr.com/creativecommons/).

PQPP releases its annotations and generated images under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Credit the PQPP authors when using the generated images and annotations.
The images in this subset have not been modified.

`assets/image-sha256.json` pins each image's contents. `assets/image-release.json`
pins the release archive, its size and SHA256. The download helper checks both
before using the images. Redistribution does not change the original licenses.
