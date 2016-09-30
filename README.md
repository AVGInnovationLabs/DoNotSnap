# DoNotSnap
<img src="https://raw.githubusercontent.com/AVGInnovationLabs/DoNotSnap/master/dns_logo.png" height="400">

An experiment in detecting DoNotSnap badges in photos, to protect privacy.  

<div style="text-align:center"><p align="center"><img src="https://raw.githubusercontent.com/AVGInnovationLabs/DoNotSnap/master/dns_badge.png" height="200"></p></div>

This program allows you to detect and identify DoNotSnap badges via a sliding-window decision tree classifier (custom heuristics are used to reduce search space).
The classifier is trained by matching samples against image templates using Affine-transform invariant SURF features.

You can find examples of using the classifier in `classify.py` and training a new classifier in `train.py`

A pre-trained classifier can be found in `classifier.pkl`
Alternative versions of the same classifier are in `classifier_alt_1.pkl` and `classifier_alt_2.pkl`
## Running classification
Run `python classify.py <path-image-to-be-tested>`
This will deserialize the classifier from `classifier.pkl` and run it on the image you supplied. A sample image could be found in `sample.jpg`
## Training your own classifier
Run `python train.py <output-file> <total-number-of-samples>`
This will read the sample filenames from `positive.txt` and `negative.txt` files.
Templates filenames are specified in `templates.txt`. A sample template could be found in `template.png`
The output is a `<output-file>.pkl` with serialized classifier.
## Dependencies
  * opencv
  * numpy
  * sklearn
  * matplotlib
  * PIL
