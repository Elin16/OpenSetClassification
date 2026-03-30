# An Open-Set Recognition and Few-Shot Learning Dataset for Audio Event Classification in Domestic Environments

This dataset is a tagged collection of pattern and unwanted sounds. Audios come from a domestic environment. The final goal of a system that can be trained with this dataset is to classify patterns sound among their classes and reject any unwanted sound (Open-set issued) plus be trined with few samples per class (Few-Shot issue).

Audio clips have been recorded with a sample rate of 16 kHz, 16 bits per samples, PCM codification and mono channel.

The number of clips are 1630 corresponding to 34 classes (24 pattern categories and 10 unwanted categories) with 40 samples per category.

In order to tackle Few-Shot learning problem in a wide range, three different configurations for training and validation are presented. These configuration are when training with 4, 2 and 1 samples per class. The number of samples correspond to the number of categories which means that in a k=4 configuration there are 40 unwanted examples (4 x 10 (number of unwanted categoires)) although they are treated as the same "class" unwanted during training.

Few-Shot learning can be approached in two manners:

* **Using prior-knowledge (transfer learning):** get a pre-trained network that have been trained with huge data (wheter labeled or not) and use it as a feature extractor or to fine-tune it

* **Use new network architectures**: when dealing with few data, cost functions and network architectures have to be modified in order to efficient optimize the classes discrimination

**TWO** baselines are presented in order to differentiate the two-lines of investigation.

* **L3 approach**: we use L3net to extract feature from the audio

* **CNN from scratch**: a CNN is trained from scratch. It is based on the DCASE 2019 Task 1C baseline.

These baselines are also analysed in different environments depending on the openness factor. This value can be calculated using the number of training classes and the number of classes during inference stage. The classes can be defined as:

* Known known class (KK): class seen in training and wanted to be targeted in inference
* Known unknown class (KU): class seen in training and NOT wanted to be targeted in inference
* Unknown unknown class (UU): class NOT seen in training and NOT wanted to be targeted in inference

The baselines are:

| Baseline | Shots | Openness | ACC KK class | ACC KU, KUU and UU classes | ACC Weighted |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Transfer learning | 1 | 0 | 13.8+-12.9 | 99.8+-1.0 | 56.8 |
| CNN | 1 | 0|  20.0+-5.0 | 99.9+-0.6 | 60.0 |
| Transfer learning | 2 | 0 | 81.1+-5.5 | 99.4+-0.8 | 90.3 |
| CNN | 2 | 0 | 17.8+-24.6 | 99.8+-0.5 | 58.8 |
| Transfer learning | 4 | 0| 94.8+-2.2 | 99.6+-0.4 | 97.2 |
| CNN | 4 | 0 | 49.5+-2.2 | 99.8+-0.3 | 74.7 |
| Transfer learning | 1 | 0.04| 57.7+-8.4 | 90.4+-5.4 84.8+-9.8 | 74.1 |
| CNN | 1 | 0.04 |  23.7+-23.5 | 98.0+-4.9 97.5+-6.6 | 60.8 |
| Transfer learning | 2 | 0.04 | 83.2+-4.8 | 90.2+-5.1 82.5+-9.6 | 86.7 |
| CNN | 2 | 0.04 | 36.8+-31.3 | 98.6+-2.3 98.0+-3.9 | 67.7 |
| Transfer learning | 4 | 0.04 | 94.3+-2.2 | 88.3+-5.7 79.4+-9.5 | 91.3 |
| CNN | 4 | 0.04 | 53.9+-38.4 | 99.5+-0.6 99.4+-0.9 | 76.7 |
| Transfer learning | 1 | 0.09 | 60.1+-7.8 | 39.6+-13.4 | 49.9 |
| CNN | 1 | 0.09 | 25.6+-21.3 | 80.9+-17.7 | 53.2 |
| Transfer learning | 2 | 0.09 | 83.3+-5.6 | 33.3+-11.6 | 58.3 |
| CNN | 2 | 0.09 | 61.7+-27.2 | 75.7+-15.6 | 68.7 |
| Transfer learning | 4 | 0.09 | 94.8+-2.4 | 26.1+-10.05 | 60.5 |
| CNN | 4 | 0.09 | 81.5+-27.6 | 80.6+-14.2 | 81.0 |

## CSVs Explanation

In [meta] folder 9 csv files are presented. Each one depends on the openness factor and the number of shots.

Each csv contains 4 columns. Let's ommit "trio_index" to make the explanation easier. First two columns are the relative path to the wav file and the label. "fold" column indicates to which k-fold the audio belongs. As it exists 40 audios per class, the number of k-fold differs due to the number of shots. Therefore, when 4 shots per class are used for training it exists 10-fold configuration. Consequently, 2 shots configuration provides a 20-fold analysis and 1 shot configuration a 40-fold (each audio example corresponds to a different fold). Take into account, that "fold" column also indicates which unwanted are used. If analyzing any "opennessmiddle.cvs" it can be seen that some unwanted examples have a k-fold in the range as indicated before and some indicates (maximum of kfolds+1). Let's see an example: "4shot_opennessmiddle.csv" (half unwanted used for training). Some unwanted have (1-10) value in "fold" column. That MUST be used for training. The ones that have 11 in "fold" column must only be used for testing. Therefore, if analyzing "4shot_opennesshigh.csv" (none unwanted used for training) all unwnated samples have 11 value in "fold" and in "4shot_opennesslow.csv" (all unwanted used for training) none sample will be indicated as 11 in "fold" columns.

In order to analysis other real-scenarios are also provided where only 3 pattern are used as KK classes. Which 3 pattern must be trained together is indicated with "trio_index" column. Pattern sounds that share same "trio_index" correspond to a trio. It exist 8 trios = 24/3. Unwanted samples do not belong to a trio because different trios are trained with same unwanteds with the corresponding k-fold configuration as explained before.

## Cite

If decide to use this dataset, please cite:

<https://arxiv.org/abs/2002.11561>

## Contact info

If you have any question related to the dataset structure of process feel free to contact:

* javier.naranjo@visualfy.com
* sergi.perez@visualfy.com

## Changelog

### v1.0.0

* Initial version of the dataset
