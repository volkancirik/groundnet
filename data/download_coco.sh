#!/bin/bash

wget https://cmu.box.com/shared/static/hrk3g9dxc279hi8lcb7wjy0cayqj3bok.json; mv hrk3g9dxc279hi8lcb7wjy0cayqj3bok.json google_refexp_train_201511_coco_aligned.json
wget https://cmu.box.com/shared/static/2gtk8smfhheidix6szhpytglrx0y74tv.json; mv 2gtk8smfhheidix6szhpytglrx0y74tv.json google_refexp_val_201511_coco_aligned.json

wget https://cmu.box.com/shared/static/o0o8btuzb4xqk8sgy7mvslexxutc82us.json; mv o0o8btuzb4xqk8sgy7mvslexxutc82us.json instances_train2014.json
wget https://cmu.box.com/shared/static/ofyqypokehn7dyqh93jf6nzthssg5vp0.json; mv ofyqypokehn7dyqh93jf6nzthssg5vp0.json instances_val2014.json

wget https://cmu.box.com/shared/static/c8lshs8tbfgkvrc4fh7gvk827p3s1mzl.json; mv c8lshs8tbfgkvrc4fh7gvk827p3s1mzl.json google_refexp_train_201511_coco_aligned_mcg_umd.json
wget https://cmu.box.com/shared/static/2fuqr1mryfrt4urvkh2bz3sq1cw3ch07.json; mv 2fuqr1mryfrt4urvkh2bz3sq1cw3ch07.json google_refexp_val_201511_coco_aligned_mcg_umd.json
