# Gender and Age Detection Using CNN

[cite_start]This project is a Deep Learning-based system that accurately detects a person's gender and age from facial images or a live webcam feed[cite: 100, 108]. [cite_start]It uses a Convolutional Neural Network (CNN) trained on the Adience dataset to classify faces into gender categories and age groups[cite: 100, 118].

## Features

* [cite_start]**Real-time Detection:** Process live video from a webcam to detect gender and age in real-time [cite: 260-261].
* [cite_start]**Image Processing:** Analyze a single image file to predict gender and age [cite: 262-263].
* [cite_start]**High Accuracy:** The model achieves a gender classification accuracy of **98.32%**, outperforming existing systems that typically reach around 96% accuracy[cite: 114, 115, 171, 172].
* [cite_start]**Age Group Classification:** Predict a person's age by classifying it into one of eight distinct age groups[cite: 215, 288].
* [cite_start]**Voice Output:** The system provides a voice note output based on the detected gender[cite: 299].

## Technologies Used

* [cite_start]**Python:** The core programming language for the project[cite: 190].
* [cite_start]**TensorFlow & Keras:** Used for building and training the Convolutional Neural Network (CNN)[cite: 335, 336, 438, 440].
* [cite_start]**OpenCV:** Utilized for image processing, face detection, and handling video streams from the webcam[cite: 190, 207].
* [cite_start]**CVLib:** A library that simplifies face detection[cite: 340, 376].
* [cite_start]**Pyttsx3:** Used to provide voice output [cite: 345-346].

## System Requirements

### Hardware
* [cite_start]**Processor:** Intel i5 (8th Gen) or equivalent[cite: 185].
* [cite_start]**RAM:** 4GB (Minimum)[cite: 186].
* [cite_start]**Internal Storage:** 5-10 GB[cite: 187].
* [cite_start]**Webcam:** At least 2.0MP[cite: 188].

### Software
* [cite_start]**Operating System:** Windows[cite: 188].
* [cite_start]**Python:** 2.7 - 3.6[cite: 190].
* **Libraries:**
    * [cite_start]`OpenCV2` [cite: 190]
    * [cite_start]`cvlib` [cite: 340]
    * [cite_start]`tensorflow` [cite: 335, 336]
    * [cite_start]`numpy` [cite: 337, 447]
    * [cite_start]`pyttsx3` [cite: 345]
    * [cite_start]`pyaudio` [cite: 344]
    * [cite_start]`scikit-learn` [cite: 445]
    * [cite_start]`matplotlib` [cite: 446]

## Project Methodology

The project's workflow is divided into four main modules:

1.  [cite_start]**Input:** Data is fed into the system either through a static image file using the `--image` tag or a live stream from a webcam[cite: 259, 262].
2.  [cite_start]**Face Detection:** The system uses a pre-trained model to detect faces in the input frame[cite: 198, 201]. [cite_start]If no face is found, it returns a "No face detected" message[cite: 301].
3.  [cite_start]**Face Processing (Classification):** The detected face is processed by a CNN[cite: 281].
    * [cite_start]**Gender Recognition:** A pre-trained gender detection model is used[cite: 206, 353].
    * [cite_start]**Age Recognition:** A separate pre-trained Caffe model (`age_net.caffemodel` and `deploy_agenet.prototxt`) is used to classify the face into one of the following eight age groups: `(0-2)`, `(4-6)`, `(8-12)`, `(15-20)`, `(25-32)`, `(38-43)`, `(48-53)`, and `(60-100)`[cite: 213, 214, 215, 288].
4.  **Output:** The system overlays the predicted gender and age on the image or video stream. [cite_start]It also provides an audio output[cite: 299, 300].

## Dataset

[cite_start]The model was trained and validated on the **Adience dataset**, which consists of 26,580 facial photographs from 2,284 distinct subjects[cite: 308, 317]. [cite_start]This dataset is known for its unconstrained images, featuring significant variations in pose, expression, lighting, and appearance, which helps in creating a robust model[cite: 314].

| Age Groups | 0-2 | 4-6 | 8-13 | 15-20 | 25-32 | 38-43 | 48-53 | 60+ | Total |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Male** | [cite_start]745 [cite: 321] | [cite_start]928 [cite: 321] | [cite_start]934 [cite: 321] | [cite_start]734 [cite: 321] | [cite_start]2308 [cite: 321] | [cite_start]1294 [cite: 321] | [cite_start]392 [cite: 321] | [cite_start]442 [cite: 321] | [cite_start]**8192** [cite: 321] |
| **Female** | [cite_start]682 [cite: 321] | [cite_start]1234 [cite: 321] | [cite_start]1360 [cite: 321] | [cite_start]919 [cite: 321] | [cite_start]2589 [cite: 321] | [cite_start]1056 [cite: 321] | [cite_start]433 [cite: 321] | [cite_start]427 [cite: 321] | [cite_start]**9411** [cite: 321] |
| **Total** | [cite_start]1429 [cite: 321] | [cite_start]2165 [cite: 321] | [cite_start]2299 [cite: 321] | [cite_start]1658 [cite: 321] | [cite_start]4907 [cite: 321] | [cite_start]2365 [cite: 321] | [cite_start]835 [cite: 321] | [cite_start]889 [cite: 321] | [cite_start]**19487** [cite: 321] |

[cite_start]_The table represents a portion of the dataset used for training[cite: 324]._

## How to Run the Project

1.  Clone the repository:
    ```
    git clone [https://github.com/vivekmelkani16/Age-Gender-Detection-using-CNNs-master.git](https://github.com/vivekmelkani16/Age-Gender-Detection-using-CNNs-master.git)
    ```
2.  Navigate to the project directory:
    ```
    cd Age-Gender-Detection-using-CNNs-master
    ```
3.  Install the required libraries:
    ```
    pip install opencv-python tensorflow numpy pyttsx3 pyaudio scikit-learn matplotlib
    ```
4.  Run the main script:

    * **For webcam input:**
        ```
        python main_code.py
        ```
    * **For image input:**
        ```
        python main_code.py --image <image_path>
        ```
        _Replace `<image_path>` with the path to your image file (e.g., `girl1.jpg`)._

## Project Structure

* `main_code.py`: The main script for running the gender and age detection system.
* `train.py`: The script used for training the gender detection model.
* `gender_detection.model`: The trained model for gender classification.
* `age_net.caffemodel`: The pre-trained Caffe model for age detection.
* `deploy_agenet.prototxt`: The network definition file for the age detection Caffe model.

## Limitations

* [cite_start]**Gender detection:** The model assumes a binary classification (male/female) which does not capture the spectrum of gender identities[cite: 1229]. [cite_start]This can lead to inaccurate classification for non-binary individuals[cite: 1230].
* [cite_start]**Age detection:** Determining age accurately from facial features alone is inherently challenging, especially for adults where changes are more subtle[cite: 1236]. [cite_start]The model classifies age into groups rather than providing an exact year[cite: 1172, 1246, 1247].
* [cite_start]**Bias:** Datasets often lack diversity in terms of ethnicity and cultural backgrounds, which can lead to biased models that perform well on some demographic groups but poorly on others[cite: 1226, 1227].

## Future Work

[cite_start]Future enhancements will focus on improving model accuracy and robustness[cite: 1293]. This includes:

* [cite_start]Combining machine learning and deep learning techniques to enhance performance[cite: 1293].
* [cite_start]Utilizing **more diverse datasets** and data augmentation techniques to reduce bias and improve generalization[cite: 1253, 1256].
* [cite_start]Exploring more advanced CNN architectures like **ResNet** or **VGG** with transfer learning[cite: 1258, 1259].
* [cite_start]Implementing **multi-task learning** to simultaneously predict age and gender for better feature sharing[cite: 1262].
* [cite_start]Incorporating **ethical considerations** by evaluating fairness across different demographics and using privacy-preserving methods[cite: 1269, 1270].
