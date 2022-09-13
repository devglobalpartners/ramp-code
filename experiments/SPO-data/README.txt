***RAMP experiments***

__Baseline configuration__
-- 20220627_baseline_aug_no_clr_v1.json: effunet model with efficientnet b0 backbone. SCCE loss function, augmentation, no CLR.


20220627
-- baseline_aug_clr_v1.json (20220627-172330): SCCE loss function, aug, cyclic learning rate determined by 20220627_baseline_augmentation_ratefinder.json. 
-- baseline_aug_clr_v2.json (20220627-214314/): SCCE loss function, aug, CLR on, initialized with best model from baseline_aug_clr_v1.json

