import streamlit as st
import pandas as pd

st.set_page_config(layout = 'wide')
col1, col2 = st.columns([3,7])
currentLoc = st.sidebar.radio('Go To', ['Home', 'Project 1','Project 2', 'Project 3', 'Project 4', 'Project 5', 'Project 6(nearing final stages)', 'Project 7(ongoing)'])
if currentLoc == 'Home':
    col2.markdown("# <center>ABDULSALAM BASIT ML PORTFOLIO</center>", unsafe_allow_html = True)
    col2.write('My public ML portfolio.\nThis portfolio includes projects majorly conputer vision projects on different areas including classification and segmentation.')
    col2.markdown("<p>Contact\nMAil:<a href = mailto: basitsalam2001@gmail.com >basitsalam2001@gmail.com </a><a href = tel: +2347050837042>+2347050837042</a></p>", unsafe_allow_html = True)
    col2.markdown("A copy of my CV can be accessed at <a href = 'https://docs.google.com/document/d/1abzvZSsUPLYUKVUf_qC6vZPpm_ByAZr3/edit?usp=sharing&ouid=109569805470530100719&rtpof=true&sd=true'> CV [gdrive link] </a>", unsafe_allow_html = True)
    col2.image('how-to-build-a-machine-learning-portfolio.jpeg')
if currentLoc == "Project 1":
    col2.markdown("# <center>ABDULSALAM BASIT ML PORTFOLIO</center>", unsafe_allow_html = True)
    col2.markdown("## <center>UNet Optimized With Differential Evolution</center>", unsafe_allow_html = True)
    col2.write('The key goal of the project was to optimize a UNet segmentation model such that we can have a small model size which will still perform as exelent as any other SoTA segmentation models.\nThe models compared with were SEGNET, PSPNET, and FCN.\nThe model achieved a weights size of about 5mb in contrast to others that are in the range of tens and hundreds mb.\nThe training data used was potato leaves downloaded from plant village which included Healthy, Early Blight and Late blight. Training was done using just early blight but models were evaluated on all 3 classes. Annotations were done manually.')
    col2.markdown("""
TOOLS USED:<ol>
                  <li> VGG IMAGE ANNOTATOR </li>
                  <li> TENSORFLOW </li>
                  <li> PYTHON </li>
                  <li> NUMPY </li>
                  <li> STREWAMLIT </li>
                  <li> OpenCV etc </li>
                  </ol>
""", unsafe_allow_html = True)
    col2.markdown("""
Project Highlight:<ul>
                  <li> Performed data engineering. </li>
                  <li> Optimized a UNet to have smmall parameters and low model size. </li>
                  <li> Achieved good metrics on the segmentation task. MIoU > 75% </li>
                  <li> Implemented a streamlit interface for visualizing predictions </li>
                  </ul>
""", unsafe_allow_html= True)
    col2.image('VIA in use.png', caption = 'VIA IN USE')

if currentLoc == "Project 2":
    df = pd.read_excel('project2 sample table.xlsx')
    col2.markdown("# <center>ABDULSALAM BASIT ML PORTFOLIO</center>", unsafe_allow_html = True)
    col2.markdown("## <center>UNet with Pretrained Encoders and  Optimized With Differential Evolution</center>", unsafe_allow_html = True)
    col2.write("""The key objectives of the project was to optimize a number of UNet segmentation architectures with pretrained encoders like VGG19 and so on with differential evolution.
               \nIn the project, some layers of he encoder of a standard Unet architecture were replaced with pretrained weights from different models.
               \nNext step was to optimize the layers of the decoder using differential evolution and record the values.
               \nThe training data used for the project include leaves of maize, cassava, yam etc.
               \nIn comparing the models, each moel is trained 6 times. A training involve training on 4 crops and evaluating on the 5th crop. The 6th training is done by combining all the crops and splitting
               \nIn the end, VGG19 model optimized at decoder layer 4 proved to be the best model on the task.""")
    col2.markdown("""
TOOLS USED:<ol>
                  <li> VGG IMAGE ANNOTATOR </li>
                  <li> Adobe photoshop (for removing unwanted noises in the images used) </li>
                  <li> TENSORFLOW </li>
                  <li> PYTHON </li>
                  <li> NUMPY </li>
                  <li> STREWAMLIT </li>
                  <li> OpenCV etc </li>
                  </ol>
""", unsafe_allow_html = True)
    col2.markdown("""
Project Highlight:<ul>
                  <li> Performed data engineering. </li>
                  <li> Replaced layers of the encoder with pretrained weights </li>
                  <li> Optimized layers of the decoders to have smmall parameters, low model size and good performance. </li>
                  <li> Achieved good metrics on the segmentation task. MIoU > 75% </li>
                  <li> Implemented a streamlit interface for visualizing predictions </li>
                  </ul>
""", unsafe_allow_html= True)
    col2.markdown('A SAMPLE OF THE REPORT AND STREAMLIT INTERFACES ARE SHOWN BELOW. Further enquiry can be checked in the <a href = "https://docs.google.com/document/d/1roWh0Jl_VcngLd_OmuGMlIP2hdXSJRbf/edit?usp=drivesdk&ouid=109569805470530100719&rtpof=true&sd=true">report [gdrive link]</a> or by contacting me', unsafe_allow_html = True)
    col2.markdown("<center>Project main report</center>", unsafe_allow_html = True)
    col2.table(df)
    col2.image('project2 sample interface.png')
    col2.markdown("<center>Sample predictions from the models</center>", unsafe_allow_html = True)
    col2.image('project2 outputs.png')

if currentLoc == "Project 4":
    df = pd.read_excel('project4 sample table.xlsx')
    col2.markdown("# <center>ABDULSALAM BASIT ML PORTFOLIO</center>", unsafe_allow_html = True)
    col2.markdown("## <center>Tiny Segmentation Network</center>", unsafe_allow_html = True)
    col2.write("""The key requirement of the project was to build a small model for segmenting plant leaves. The model should be small enough to be deployed a a microcontroller but also powerful enough to perform up to task.
               \nIn the project, the model mimicked a standard unet but the approach used was utilizing a deeper model and creating different decoder networks for each class and then combining the outputs at the inference.
               \nThe training data used for the project include leaves of maize, cassava, yam etc.,all featuring different types of diseases.
               \nFor evaluation, the model was first evaluated against othe standard Unet architectures with pretrained encoders and then with other SoTA segmentation models.
               \nIn the end, the model had a total model size of 8mb and it performed the best on the task.""")
    col2.markdown("""
TOOLS USED:<ol>
                  <li> VGG IMAGE ANNOTATOR </li>
                  <li> Adobe photoshop (for removing unwanted noises in the images used) </li>
                  <li> TENSORFLOW </li>
                  <li> PYTHON </li>
                  <li> NUMPY </li>
                  <li> STREWAMLIT </li>
                  <li> OpenCV etc </li>
                  <li> FLUTTER </li>
                  </ol>
""", unsafe_allow_html = True)
    col2.markdown("""
Project Highlight:<ul>
                  <li> Performed data engineering. </li>
                  <li> Created a tiny segmentation model.</li>
                  <li> Achieved good metrics on the segmentation task. MIoU > 80% </li>
                  <li> Implemented a streamlit interface for visualizing predictions </li>
                  <li> Deloyed The model on an android phone
                  </ul>
""", unsafe_allow_html= True)
    col2.markdown('A SAMPLE OF THE REPORT AND OTHER MODEL VISUALIZATIONS ARE SHOWN BELOW. Further enquiry can be checked in the <a href = "https://docs.google.com/document/d/1ZuVrWaqN7Jhi-rECwiz1xBuiiiA-LPCU/edit?usp=drivesdk&ouid=103980037459622424907&rtpof=true&sd=true">report [gdrive link]</a> or by contacting me', unsafe_allow_html = True)
    col2.markdown("<center>Project main report</center>", unsafe_allow_html = True)
    col2.table(df)
    col2.markdown("<center>Sample predictions from the models</center>", unsafe_allow_html = True)
    col2.image('project4 outputs.png')
    col2.markdown("<center>Model visualization</center>", unsafe_allow_html = True)
    col3, col4 = col2.columns([5,5])
    col3.image('project4 novel outlook.jpg', caption = 'model outlook')
    col4.image('project4 NOVEL MODEL ARCHITECTURE.png', caption = 'model arch')

    col2.markdown("<center>Android App</center>", unsafe_allow_html = True)
    col2.image('segmenterApp.jpg', caption = 'Android deployment')

if currentLoc == "Project 5":
    df = pd.read_excel('project5 sample table.xlsx')
    col2.markdown("# <center>ABDULSALAM BASIT ML PORTFOLIO</center>", unsafe_allow_html = True)
    col2.markdown("## <center>Tiny Segmentation and Classification Network</center>", unsafe_allow_html = True)
    col2.write("""The goal of the project was to build a small model for segmenting and classifying plant leaves. The model should be small enough to be deployed a a microcontroller but also powerful enough to perform up to task.
               \nIn the project, a classification model was built and then integrated with the segmentation model built in the prior project.
               \nIn building the classification model, some concepts used were
               <ul>
               <li> THE USE OF WIDE NETWORKS </li>
               <li> THE USE OF DEEP NETWORKS </li>
               <li> THE USE OF INCEPTION MODULES </li>
               <li> THE USE OF RESIDUAL CONNECTIONS </li>
               <li> THE USE OF 1x1 CONVOLUTIONS </li>
               <li> THE USE OF CNN INSTEAD OF DENSE NETWORKS </li>
               </ul>
               \nThe training data used for the project include leaves of maize, cassava, yam etc.,all featuring different types of diseases as in the case of the segmentation model.
               \nFor evaluation, the classification model was evaluated against some other SoTA classification models.
               \nIn the end, the classification model had a total model size of <400kb and it performed the best on the task but second best to Densenet pretrained model.""", unsafe_allow_html = True)
    col2.markdown("""
TOOLS USED:<ol>
                  <li> VGG IMAGE ANNOTATOR </li>
                  <li> Adobe photoshop (for removing unwanted noises in the images used) </li>
                  <li> TENSORFLOW </li>
                  <li> PYTHON </li>
                  <li> NUMPY </li>
                  <li> STREWAMLIT </li>
                  <li> OpenCV etc </li>
                  <li> FLUTTER </li>
                  </ol>
""", unsafe_allow_html = True)
    col2.markdown("""
Project Highlight:<ul>
                  <li> Performed data engineering. </li>
                  <li> Created a tiny classifcation model.</li>
                  <li> Achieved good metrics on the segmentation task. F1 score > 98% </li>
                  <li> Implemented a streamlit interface for visualizing predictions </li>
                  <li> Deloyed The model on an android phone
                  </ul>
""", unsafe_allow_html= True)
    col2.markdown('A SAMPLE OF THE REPORT AND OTHER MODEL VISUALIZATIONS ARE SHOWN BELOW. Further enquiry can be checked in the <a href = "https://docs.google.com/document/d/1X0QJ3VesoOagpyS5ptqLvy3sjBW9Tb18/edit?usp=drivesdk&ouid=103980037459622424907&rtpof=true&sd=true">report [gdrive link]</a> or by contacting me', unsafe_allow_html = True)
    col2.markdown("<center>Project main report</center>", unsafe_allow_html = True)
    col2.table(df)
    
    col2.image('project5 sample interface.png')
    col2.markdown("<center>Model visualization</center>", unsafe_allow_html = True)
    col3, col4 = col2.columns([5,5])
    col3.image('project5 large.png', caption = 'model large')
    col4.image('project5 small.png', caption = 'model small')

    col2.markdown("<center>Android App</center>", unsafe_allow_html = True)
    col2.image('segClassApp.jpg', caption = 'Android deployment')
if currentLoc == 'Project 6(nearing final stages)':
    col2.markdown("# <center>ABDULSALAM BASIT ML PORTFOLIO</center>", unsafe_allow_html = True)
    col2.markdown("## <center>ARDUINO/ML AFib DETECTION</center>", unsafe_allow_html = True)
    col2.write("""AN ECG ACQUISITION SYSTEM. The raw ECG is taken using AD8232 sensor with Arduino at 500Hz sampling frequency and stored in an array. Peaks are obtained from the array and features extracted. The extracted features are fed into a RandomForestClassifier model already built using scikit learn and converted to C++ using micromlgen python library.
             \nThe model was trained to detect Normal Sinus Rhythm, AFib and Other Heart diseases. The model prediction will be displayed on an LCD screen. The ecg array and extracted features will be saved on SD card and can be used on a streamlit app to display some information.
             Anticipate!!!
             """)

if currentLoc == 'Project 7(ongoing)':
    col2.markdown("# <center>ABDULSALAM BASIT ML PORTFOLIO</center>", unsafe_allow_html = True)
    col2.markdown("## <center>NEURAL TRANSLATION AND AUDIO GENERATION</center>", unsafe_allow_html = True)
    col2.write('ANTICIPATE!!!. \nIt is a system that takes audio directly from a microphone, transcribes it, translate to other languages, trasforms the translated texts to audio, send over bluetooth with raspberry to multiple bluetooth speakers with each speaker handling specific language.')

if currentLoc == "Project 3":
    col2.markdown("# <center>ABDULSALAM BASIT ML PORTFOLIO</center>", unsafe_allow_html = True)
    col2.markdown("## <center>SCIZOPHRENIA DETECTION MODEL</center>", unsafe_allow_html = True)
    col2.write("""The main goal of the project is to develop a schizophrenia detection model with EfficientNet.
               \nThe data used was retrieved from schizconnect and processed using matlab dpabi/dparsf and SPI""")
    col2.markdown('Further enquiry can be checked in the <a href = "https://docs.google.com/document/d/1-2tQln0dTxbZkHM7kayoc9dcnRcICnUD/edit?usp=drivesdk&ouid=109569805470530100719&rtpof=true&sd=true">report [gdrive link]</a> or by contacting me', unsafe_allow_html = True)
    col2.markdown("""
TOOLS USED:<ol>
                  <li> SCHIZOPHRENIA </li>
                  <li> OPENCV</li>
                  <li> TENSORFLOW </li>
                  <li> PYTHON </li>
                  <li> NUMPY </li>
                  <li> SimpleITK </li>
                  <li> DLTK</li>
                  <li> MATLAB</li>
                  <li> DPABI</li>
                  </ol>
""", unsafe_allow_html = True)
    col2.markdown("""
Project Highlight:<ul>
                  <li> Performed data engineering. </li>
                  <li> PROCESSED MRI DATA </li>
                  <li> TRAINED EFFICIENTNET MODELS</li>
                  </ul>
""", unsafe_allow_html= True)
    
    col2.markdown("<center>Project Visualizations</center>", unsafe_allow_html = True)
    col2.image('project3 data collection.png', caption = 'DATA ACQISITION')
    col2.image('project3 dpabi proessing.png', caption = 'DPABI PROCESSING')
    col2.image('project3 processed images.png', caption = 'PROCESSED IMAGES')
