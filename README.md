# Handwriting Recognition in Pytorch

The goal of this project is to first segment images of page scans into
handwriting regions, then pass them to a sequential model that transcribes each
line into words.

## Setup

Download the [IAM dataset](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database)
and extract to `/[project]/data` directory. Put each set of files in their own
directory, ie `data/xml`, `data/forms` etc.

Run `python data.py` to generate CSVs for page segmentations.

(To be continued)
