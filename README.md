# Gender and Age Detection Using CNN

This project is a Deep Learning-based system that accurately detects a person's gender and age from facial images or a live webcam feed. It uses a Convolutional Neural Network (CNN) trained on the Adience dataset to classify faces into gender categories and age groups.

## Features

* **Real-time Detection:** Process live video from a webcam to detect gender and age in real-time.
* **Image Processing:** Analyze a single image file to predict gender and age.
* **High Accuracy:** The model achieves a gender classification accuracy of **98.32%**, outperforming existing systems that typically reach around 96% accuracy.
* **Age Group Classification:** Predict a person's age by classifying it into one of eight distinct age groups.
* **Voice Output:** The system provides a voice note output based on the detected gender.

## Technologies Used

* **Python:** The core programming language for the project.
* **TensorFlow & Keras:** Used for building and training the Convolutional Neural Network (CNN).
* **OpenCV:** Utilized for image processing, face detection, and handling video streams from the webcam.
* **CVLib:** A library that simplifies face detection.
* **Pyttsx3:** Used to provide voice output.

## System Requirements

### Hardware
* **Processor:** Intel i5 (8th Gen) or equivalent.
* **RAM:** 4GB (Minimum).
* **Internal Storage:** 5-10 GB.
* **Webcam:** At least 2.0MP.

### Software
* **Operating System:** Windows.
* **Python:** 2.7 - 3.6.
* **Libraries:**
    * `OpenCV2`
    * `cvlib`
    * `tensorflow`
    * `numpy`
    * `pyttsx3`
    * `pyaudio`
    * `scikit-learn`
    * `matplotlib`

## Project Methodology

The project's workflow is divided into four main modules:

1.  **Input:** Data is fed into the system either through a static image file using the `--image` tag or a live stream from a webcam.
2.  **Face Detection:** The system uses a pre-trained model to detect faces in the input frame. If no face is found, it returns a "No face detected" message.
3.  **Face Processing (Classification):** The detected face is processed by a CNN.
    * **Gender Recognition:** A pre-trained gender detection model is used.
    * **Age Recognition:** A separate pre-trained Caffe model (`age_net.caffemodel` and `deploy_agenet.prototxt`) is used to classify the face into one of the following eight age groups: `(0-2)`, `(4-6)`, `(8-12)`, `(15-20)`, `(25-32)`, `(38-43)`, `(48-53)`, and `(60-100)`.
4.  **Output:** The system overlays the predicted gender and age on the image or video stream. It also provides an audio output.

## Dataset

The model was trained and validated on the **Adience dataset**, which consists of 26,580 facial photographs from 2,284 distinct subjects. This dataset is known for its unconstrained images, featuring significant variations in pose, expression, lighting, and appearance, which helps in creating a robust model.

| Age Groups | 0-2 | 4-6 | 8-13 | 15-20 | 25-32 | 38-43 | 48-53 | 60+ | Total |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Male** | 745 | 928 | 934 | 734 | 2308 | 1294 | 392 | 442 | **8192** |
| **Female** | 682 | 1234 | 1360 | 919 | 2589 | 1056 | 433 | 427 | **9411** |
| **Total** | 1429 | 2165 | 2299 | 1658 | 4907 | 2365 | 835 | 889 | **19487** |

_The table represents a portion of the dataset used for training._

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

* **Gender detection:** The model assumes a binary classification (male/female) which does not capture the spectrum of gender identities. This can lead to inaccurate classification for non-binary individuals.
* **Age detection:** Determining age accurately from facial features alone is inherently challenging, especially for adults where changes are more subtle. The model classifies age into groups rather than providing an exact year.
* **Bias:** Datasets often lack diversity in terms of ethnicity and cultural backgrounds, which can lead to biased models that perform well on some demographic groups but poorly on others.

## Future Work

Future enhancements will focus on improving model accuracy and robustness. This includes:

* Combining machine learning and deep learning techniques to enhance performance.
* Utilizing **more diverse datasets** and data augmentation techniques to reduce bias and improve generalization.
* Exploring more advanced CNN architectures like **ResNet** or **VGG** with transfer learning.
* Implementing **multi-task learning** to simultaneously predict age and gender for better feature sharing.
* Incorporating **ethical considerations** by evaluating fairness across different demographics and using privacy-preserving methods.
