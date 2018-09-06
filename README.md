# Deep Transfer Learning for Art Classification Problems

We hereby release the code and models of the paper "Deep Transfer Learning for Art Classification Problems", which will be presented at the ECCV VisArt Workshop on Computer Vision for Art Analysis (September 2018, Munich GE).

We investigate the performances of different Deep Convolutional Neural Networks (DCNNs) which pre-trained on the ImageNet dataset, aim to tackle 3 different art classification problems. 

![alt tag](https://user-images.githubusercontent.com/14283557/44711771-6d076500-aaaf-11e8-8c7f-c09e66fe7825.png)

The contributions of our work can be replicated as follows:

  * If you aim to investigate the performances of DCNNs which have been pretrained on the ImageNet dataset only and can either be fine tuned, or used as off the shelf feature extractors for a separetly trained softmax classifier, you can find the code for this set of experiments in `./transfer_learning_experiment/from_imagenet_to_art/`. Assuming you have a folder containing images in `*.jpg` format and a `*.csv` file representing the relative metadata like in the toy example presented in `./metadata/`, just fill the `job.sh` file with the appropriate paths and which TL mode you would like to explore: then run `./job.sh`. The script will create the appropriate Training-Validation and Testing Sets, store them in `hdf5` format and call the ResNet architecture.
  
  * If you aim to use a DCNN which has been fine-tuned on a large artistic collection first and investigate its TL performances on a smaller artistic dataset you can find the appropriate code in `./transfer_learning_experiment/from_one_art_to_another/` and the already trained models in `./models/`. The pipeline for testing these kind of DCNNs is exactly the same one as the one presented in the previous step but a better "Artistic DCNN" will be used instead of a standard ImageNet pretrained model (which in case of the `VGG19` architecture has been renamed as `RijksVGG19Net`). 
  
  * If you would like to explore which saliency maps get activated in a DCNN when classifying a particular artistic image and compare how such activation maps change between differently trained networks you can use the code in `./saliency_maps_activations/` and the `deep_viz.py` script. It will use the images present in `./figures/` and you will be able to obtain the following Figure
  
  ![alt tag](https://user-images.githubusercontent.com/14283557/44716330-4ac71480-aaba-11e8-8e08-49bb7153493e.jpg)
  
  * You can download the images of the Rijksmuseum from [here](https://staff.fnwi.uva.nl/t.e.j.mensink/uva12/rijks/) while you can contact me directly for the ones of the Antwerp dataset. I will also be happy to share the splits with you!
  
  Lastly you can find a version of the ECCV paper in `./paper`.
  
 **Do not hesitate to submit any issues and contact me if you are open for collaborations in the field of Deep Leaning applied to Art**
