# Repository Title Goes Here

Culminating our learning on Andrew NGs deep learning specialization left us with an abundance of knowledge in a few popular fields of deep learning. While we understood the fundamentals, we were actively looking for ways to deepen our understanding of Convolution Neural Networks (CNNs). An interest in applying popular object detection models on the popular show ‘The Office’ led us to creating our first ever deep learning project. 
Here, we used the yolov3 model and trained it on our custom Office dataset obtained through web scraping of google images. We picked four popular characters from the show as our four classes and trained the model on google collab. 


## Table of Contents


- [Creating The Custom Dataset](#creating-the-custom-dataset)
- [Testing a Pre-trained YOLO Model](#testing-a-pre-trained-yolo-model)
- [Training the Dataset Using YOLOv3 on Google Colab](#training-the-dataset-using-yolov3-on-google-colab)
- [Running YOLOv3 With Custom Weights For Live Object Detection](#running-yolov3-with-custom-weights-for-live-object-detection)

---

## Creating the custom dataset

The first step with creating your own dataset is to gather a large number of images, preferably 200 images per class. If you are lucky, you might find a dataset for your project on the internet. If not, you can always go down the traditional path of clicking your own pictures or downloading images from the internet. 

For the latter, it is a lot easier to have a piece of code that can automatically download images for your tasks and categorise them into different classes. ‘Scraping’ does just the job. 

To implement the code and get it running for your task, [Image Scraping with Python](https://towardsdatascience.com/image-scraping-with-python-a96feda8af2d) does an excellent job explaining the process of web scraping google images. To implement the same

> Install chrome driver and place it in the same folder as scraping.ipynb

Change the following lines of code

```shell
 DRIVER_PATH = r‘Your Chrome Driver Path’
```
```shell
 def search_and_download(search_term:str,driver_path:str,target_path=r'Your Folder Path',number_images=5):
```
```shell
 search_term = ‘The Class You Want to Scrape From the Internet’
```
```shell
 driver_path = r‘Your Chrome Driver Path’
```
```shell
 search_and_download(search_term=search_term,driver_path=driver_path,number_images=Number of images you want to scrape)
```

While the code does the job pretty well, there are times when the number of images you choose to scrape might exceed the number of search results.

![](/README_Img/1.png)

If this happens, stop the code from running, change the number of images to the latest number of search results found (Here it is 412) and then re-run the code. 

While this cut shorts your time of having to manually download images off the web, several of the images in your class folder will contain images that do not belong to your class. This is where the tedious process of filtering unrequired images comes into play. 

Here, you manually go through your images and delete the ones that are not your class. Once this is done, you will have to manually annotate your images. To do this

> Install labelImg
> Go to windows_v1.8.0\windows_v1.8.0\data and change the default classes in predefined_classes to the names of your classes 

![](/README_Img/2.png)

Next open labelImg and do the following

- Open dir and select the folder that contains the images of your class
- Open dir again and select the same folder to add your image annotations
- For yolov3, change PascalVOC to YOLO. This saves the annotation in .txt instead of .xml
- Select a picture from the file list, click on Create RectBox and draw the bounding box around the class 
- Then click on the class in the class list

![](/README_Img/3.png)

Repeat this process for all your images. Once you have completed the process, compile all the class images and annotations in one folder and store it as a zip file. 

---

## Testing a Pre-trained Yolo model

We used [How to Perform Object Detection With YOLOv3 in Keras](https://machinelearningmastery.com/how-to-perform-object-detection-with-yolov3-in-keras/) to run test on their yolo model to make sure we had a model that ran error free before we trained it for our own dataset. 
The process is relatively simple

- Save the pre-trained weights in the same folder as the source code
- Download the source code from the website and run it to add the weights to the model
- After the model.h5 file is created, download an image of any of the classes listed in labels in the source code
- Change the following

```shell
 photo_filename = 'Name of Your File' 
```
- Run the code and see if the model has currently identified the class

If everything runs error free, you have a yolov3 model trained on the coco dataset and are ready to move on to training the model on your custom dataset.

### Training the Dataset Using YOLOv3 on Google Colab

- Used The AI Guy’s Google Colab notebook [YOLOv3_Tutorial](https://colab.research.google.com/drive/1Mh2HP_Mfxoao6qNFbhfV3u28tG8jAVGk). After enabling the GPU on the colab notebook in the repository, it is necessary to clone darknet from AlexeyAB’s famous repository. Enable OpenCV and GPU for darknet.
- Images can also be uploaded to your Google Drive and easily have detections run on them. Rename the folder with the annotated images on the local machine to ‘obj’. Zip the folder and upload it to Google Drive. 
- After creating a separate folder, specifically for the colab files, mounting the drive onto the cloud VM is required to be able to access its contents. 

To configure the files for training

- Create a custom cfg file – set batch = 64 and subdivision =16. Change the last conv layer before every yolo layer to match the number of classes.  4 in our case. Change the number of filters  f = num_of_anchors/3 * (classes + 1 + 4)
- Used a text editor to create an obj.names file

![](/README_Img/4.png)

- Also created an obj.data file and filled it in like this

![](/README_Img/5.png)

This backup path is where we will save the weights to of our model throughout training. Create a backup folder in your google drive and put its correct path in this file.
- To create the train.txt we used a script that generates it for us. Download the script from the github repo and upload it onto your drive.

Used pre-trained weights for the convolutional layers, for it to be more accurate and not have to train for as long. 
Final step involved training the custom object detector for about 6-7 hours, achieving a loss of under 0.1. The weights after every 1000 epochs were saved automatically in the weights folder created on our google drive.


### Running YOLOv3 With Custom Weights For Live Object Detection

Used The AI Guy’s GitHub repository [Object-Detection-API](https://github.com/theAIGuysCode/Object-Detection-API) for this task. 
All of the tasks were performed in the command prompt itself. First step was to clone the above repository. 

```shell
 git clone https://github.com/theAIGuysCode/Object-Detection-API
```
After setting the current working directory to the above folder downloaded on your local machine, we created an Anaconda environment. We also used the GPU version of YOLOv3.

```shell
 conda env create -f conda-gpu.yml
 conda activate yolov3.gpu
```
Add your custom weights file to the weights folder and your custom .names file into the data/labels folder. 
Now we need to convert the yolov3 weights into Tensorflow .ckpt model files

```shell
 python load_weights.py
```
There will be a few packages which may or may not be installed on your system. Hence every time you run into a package missing error, it is necessary to ‘pip install’ said package.
Open the detect_video.py file in any text editor and make changes to the number of classes. 
Finally, we are ready to make detections on a video clip. Place your clip in the data/video folder and run the following command.

```shell
 python detect_video.py –video path_to_file.mp4 –output ./detections/output.avi
```
You can find the final output video in the detections folder!
