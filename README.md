# RAMP starting kit on GAN Anime


> Authors: MAI Huu Tan, CARON Marceau, SALOMON Yohann, LIU Annie, BERTHOLOM François & BIGOT Alexandre


#### Set up

Open a terminal and

1. install the `ramp-workflow` library (if not already done)
  ```
  $ pip install ramp-workflow
  ```
  
2. Follow the ramp-kits instructions from the [wiki](https://github.com/paris-saclay-cds/ramp-workflow/wiki/Getting-started-with-a-ramp-kit)

#### Local notebook

Get started on this RAMP with the [dedicated notebook]().

To test the starting-kit, run


```
ramp-test --quick-test
```


#### Help
Go to the `ramp-workflow` [wiki](https://github.com/paris-saclay-cds/ramp-workflow/wiki) for more help on the [RAMP](https://ramp.studio) ecosystem.


---

## TODO

### [`problem.py` file](https://paris-saclay-cds.github.io/ramp-docs/ramp-workflow/stable/problem.html#problem)

- Implem custom [Prediction type](https://github.com/paris-saclay-cds/ramp-workflow/tree/master/rampwf/prediction_types)
- Implem the [workflow](https://github.com/paris-saclay-cds/ramp-workflow/tree/master/rampwf/workflows)
- Implem score metrics
- specify cross-validation ?
- get_train_data() and get_test_data() ?

### build submissions/starting_kit

- implem simple auto-encoder
