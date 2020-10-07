#!/bin/bash
wget https://cmu.box.com/shared/static/74ujua5gz6sglwx7kt3160oc1c6cgcw8.gz; mv 74ujua5gz6sglwx7kt3160oc1c6cgcw8.gz refcocog_sentence.tar.gz
tar xvfz refcocog_sentence.tar.gz

wget https://cmu.box.com/shared/static/4r0tlvtw1tyd0xgi4qnifcsa04i3bp2c.gz; mv 4r0tlvtw1tyd0xgi4qnifcsa04i3bp2c.gz google_ref_val_images.tar.gz
tar xvfz google_ref_val_images.tar.gz

wget https://cmu.box.com/shared/static/ixf6y9jy8m6r4eti1e89p62r29oz23v2.gz; mv ixf6y9jy8m6r4eti1e89p62r29oz23v2.gz google_ref_dev_images.tar.gz
tar xvfz google_ref_dev_images.tar.gz


wget https://cmu.box.com/shared/static/f2q9i7a72y36w6hwuhocgovgvum4wjea.h5;mv f2q9i7a72y36w6hwuhocgovgvum4wjea.h5 stanford_cmn_refcocog_iou05.box%2Bmeta%2Bsmax.h5
wget https://cmu.box.com/shared/static/ozmzgk0wchyn13mwvckmwwe04t5ygslj.pkl;mv ozmzgk0wchyn13mwvckmwwe04t5ygslj.pkl stanford_cmn_refcocog_iou05.NOtriplets.pkl
wget https://cmu.box.com/shared/static/ye79uzx1rhsvos97jl674ulhp4x736az.pkl;mv ye79uzx1rhsvos97jl674ulhp4x736az.pkl stanford_cmn_refcocog_iou05.triplets.pkl

wget https://cmu.box.com/shared/static/ilffctq60n3p1grq7579ktr1gbyq2b6i.pkl;mv ilffctq60n3p1grq7579ktr1gbyq2b6i.pkl stanford_cmn_refcocog_iou05_umd.triplets.pkl
wget https://cmu.box.com/shared/static/ajaoiwa2dtytxffsss85yv4m81getent.pkl;mv ajaoiwa2dtytxffsss85yv4m81getent.pkl stanford_cmn_refcocog_iou05_umd.NOtriplets.pkl

wget https://cmu.box.com/shared/static/4ci9lkcc0fqmvxhrfiz9bvg1yxzhd51v.glove;mv 4ci9lkcc0fqmvxhrfiz9bvg1yxzhd51v.glove wordvec.glove

wget https://cmu.box.com/shared/static/yd2qwkzkzs8tpdk6v0d7lvo2cvgln7tf.npy; mv yd2qwkzkzs8tpdk6v0d7lvo2cvgln7tf.npy refcocog.trn.npy
wget https://cmu.box.com/shared/static/v3nom6jw63z7padz7fgxmvofnz3pvckv.npy; mv v3nom6jw63z7padz7fgxmvofnz3pvckv.npy refcocog.val.npy
