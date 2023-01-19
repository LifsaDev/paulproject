import streamlit as st
import imageio
import matplotlib.pyplot as plt
import pandas as pd 
from pathlib import Path  
import glob
import pickle
from scipy import ndimage as nd
import cv2
from PIL import Image
from skimage import io,data,img_as_ubyte
from sklearn.cluster import KMeans,DBSCAN,FeatureAgglomeration,spectral_clustering
from skimage.filters import  threshold_mean, threshold_triangle,threshold_isodata,threshold_sauvola,threshold_li,threshold_minimum,threshold_yen,threshold_niblack
from streamlit_option_menu import option_menu
from skimage import io,data,img_as_ubyte
from skimage.filters import threshold_multiotsu
import numpy as np
from fcmeans import FCM
from skimage.filters import sobel, prewitt, roberts, scharr 
from streamlit_lottie import st_lottie
import requests
import json

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)
def threshoding(f,L):
    # create a new image with zeros
    f_tr = np.ones(f.shape).astype(np.uint8)
    # setting to 0 the pixels below the threshold
    f_tr[np.where(f<L)] = 0
    return f_tr

def result_show(img,output,modele,thresholds):
    
    fig,ax = plt.subplots(nrows=1,ncols=3,figsize=(10,3.5))

    ax[0].imshow(img,cmap='gray')
    ax[0].set_title('original')
    ax[0].axis("off")

    ax[1].hist(img.ravel(),bins=255)
    ax[1].set_title('Histogram')
    for thresh in thresholds:
        ax[1].axvline(thresh,color = "r")

    
    ax[2].imshow(output,cmap='Accent')
    ax[2].set_title(modele)
    ax[2].axis("off")
    st.pyplot(fig)

def result_show_c(img,output,modele,thresholds):
    
    fig,ax = plt.subplots(nrows=1,ncols=3,figsize=(10,3.5))

    ax[0].imshow(img,cmap='gray')
    ax[0].set_title('original')
    ax[0].axis("off")

    ax[1].hist(img.ravel(),bins=255)
    ax[1].set_title('Histogram')
    
    ax[1].axvline(thresholds,color = "r")


    ax[2].imshow(output,cmap='Accent')
    ax[2].set_title(modele)
    ax[2].axis("off")
    st.pyplot(fig)

def result_show_r(img,output,modele):


    fig,ax = plt.subplots(nrows=1,ncols=3,figsize=(10,3.5))

    ax[0].imshow(img,cmap='gray')
    ax[0].set_title('original')
    ax[0].axis("off")

    ax[1].hist(img.ravel(),bins=255)
    ax[1].set_title('Histogram')
    # for thresh in thresholds:
    #     ax[1].axvline(thresh,color = "r")
 
    ax[2].imshow(output,cmap='Accent')
    ax[2].set_title(modele)
    ax[2].axis("off")
    st.pyplot(fig)

def feature_extraction(img):
    df = pd.DataFrame()


#All features generated must match the way features are generated for TRAINING.
#Feature1 is our original image pixels
    img2 = img.reshape(-1)
    df['Original Image'] = img2

#Generate Gabor features
    num = 1
    kernels = []
    for theta in range(2):
        theta = theta / 4. * np.pi
        for sigma in (1, 3):
            for lamda in np.arange(0, np.pi, np.pi / 4):
                for gamma in (0.05, 0.5):
#               print(theta, sigma, , lamda, frequency)
                
                    gabor_label = 'Gabor' + str(num)
#                    print(gabor_label)
                    ksize=9
                    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)    
                    kernels.append(kernel)
                    #Now filter image and add values to new column
                    fimg = cv2.filter2D(img2, cv2.CV_8UC3, kernel)
                    filtered_img = fimg.reshape(-1)
                    df[gabor_label] = filtered_img  #Modify this to add new column for each gabor
                    num += 1
########################################
#Geerate OTHER FEATURES and add them to the data frame
#Feature 3 is canny edge
    edges = cv2.Canny(img, 100,200)   #Image, min and max values
    edges1 = edges.reshape(-1)
    df['Canny Edge'] = edges1 #Add column to original dataframe

    from skimage.filters import roberts, sobel, scharr, prewitt

#Feature 4 is Roberts edge
    edge_roberts = roberts(img)
    edge_roberts1 = edge_roberts.reshape(-1)
    df['Roberts'] = edge_roberts1

#Feature 5 is Sobel
    edge_sobel = sobel(img)
    edge_sobel1 = edge_sobel.reshape(-1)
    df['Sobel'] = edge_sobel1

#Feature 6 is Scharr
    edge_scharr = scharr(img)
    edge_scharr1 = edge_scharr.reshape(-1)
    df['Scharr'] = edge_scharr1

    #Feature 7 is Prewitt
    edge_prewitt = prewitt(img)
    edge_prewitt1 = edge_prewitt.reshape(-1)
    df['Prewitt'] = edge_prewitt1

    #Feature 8 is Gaussian with sigma=3
    from scipy import ndimage as nd
    gaussian_img = nd.gaussian_filter(img, sigma=3)
    gaussian_img1 = gaussian_img.reshape(-1)
    df['Gaussian s3'] = gaussian_img1

    #Feature 9 is Gaussian with sigma=7
    gaussian_img2 = nd.gaussian_filter(img, sigma=7)
    gaussian_img3 = gaussian_img2.reshape(-1)
    df['Gaussian s7'] = gaussian_img3

    #Feature 10 is Median with sigma=3
    median_img = nd.median_filter(img, size=3)
    median_img1 = median_img.reshape(-1)
    df['Median s3'] = median_img1

    #Feature 11 is Variance with size=3
    variance_img = nd.generic_filter(img, np.var, size=3)
    variance_img1 = variance_img.reshape(-1)
    df['Variance s3'] = variance_img1  #Add column to original dataframe


    return df


def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
st.markdown("""
<style>
.first_titre {
    font-size:60px !important;
    font-weight: bold;
    box-sizing: border-box;
    text-align: center;
    width: 100%;
}
.intro{
    text-align: justify;
    font-size:20px !important;
}
.grand_titre {
    font-size:20px !important;
    font-weight: bold;
    text-align: center;
    # text-decoration: underline;
    text-decoration-color: #4976E4;
    text-decoration-thickness: 5px;
}
.section{
    font-size:20px !important;
    font-weight: bold;
    text-align: center;
    text-decoration: underline;
    text-decoration-color: #111111;
    text-decoration-thickness: 3px;
}
.petite_section{
    font-size:16px !important;
    font-weight: bold;
}
.nom_colonne_page3{
    font-size:17px !important;
    text-decoration: underline;
    text-decoration-color: #000;
    text-decoration-thickness: 1px;
}
.headtitle{
    color: #8B0000;
    font-weight: bold;
}
.menu-title {
    color: #8B0000;
}

</style>
""", unsafe_allow_html=True)
# head = st.markdown(' <p class="headtitle">Segmentation </p> Platform', unsafe_allow_html=True)

choose = option_menu("Segmentation Platform",["Home","Seuillage","Contour","Region","Forme"],
    icons=['house','bi bi-bar-chart-line-fill','bi bi-bounding-box-circles','bi bi-pie-chart-fill','bi bi-x-diamond-fill'],
    menu_icon = "None", default_index=0,
    styles={
        "container": {"padding": "5!important", "background-color": ""},
        "icon": {"color": "orange", "font-size": "18px"}, 
        "nav-link": {"font-size": "10px", "text-align": "left", "margin":"5px", "--hover-color": ""},
        "nav-link-selected": {"background-color": ""},
    },orientation = "horizontal"
    )
seuil_modele = ["Multi Otsu","Li","Yen","Iso data","Triangle","Mean","Median"]


# *******************************seuillage***************************************
if choose=="Seuillage":

    kindmodele = ""
    with st.sidebar:
        st.markdown("## **1. Import Image :open_file_folder:** ##") 
        datafile = st.file_uploader("Upload Image",type=None)
        st.markdown("## **2. Image Overwiew:mag:** ##") 
        if datafile is not None:
            st.write(type(datafile))
            file_details = {"filename":datafile.name,"filetype":datafile.type,"filesize":datafile.size}
            st.write(file_details)
            img = Image.open(datafile)
            img.save("img.png","png")

            # data = (datafile)
        
        st.markdown("## **3. Choose technic :computer:** ##") 
        kindmodele = st.sidebar.selectbox("What technic?", seuil_modele)
        

    if kindmodele=="Multi Otsu":
        # col1,col2 = st.columns(2)
    
        classe = st.number_input("Numbers of classe", min_value=2, max_value=10, value=2)
        img = imageio.imread("img.png")
        thresholds = threshold_multiotsu(img,classes=int(classe))

        print(thresholds)
        regions = np.digitize(img,bins=thresholds)
        output = img_as_ubyte(regions)
        st.title("Result of " +kindmodele + "   technic")
        result_show(img,output,kindmodele,thresholds)
       # st.download_button('Download Image', thresholds)    
    if kindmodele=="Li":
        img = imageio.imread("img.png")
        s = threshold_li(img)
        img_seg=  threshoding(img,s)
        st.title("Result of " +kindmodele + "   technic")
        result_show_c(img,img_seg,kindmodele,s)
    
    if kindmodele=="Yen":
        img = imageio.imread("img.png")
        s = threshold_yen(img)
        img_seg=  threshoding(img,s)
        st.title("Result of " +kindmodele + "   technic")
        result_show_c(img,img_seg,kindmodele,s)
    
    if kindmodele=="Iso data":
        img = imageio.imread("img.png")
        s = threshold_isodata(img)
        img_seg=  threshoding(img,s)
        st.title("Result of " +kindmodele + "   technic")
        result_show_c(img,img_seg,kindmodele,s)
    
    if kindmodele=="Triangle":
        img = imageio.imread("img.png")
        s = threshold_triangle(img)
        img_seg=  threshoding(img,s)
        st.title("Result of " +kindmodele + "   technic")
        result_show_c(img,img_seg,kindmodele,s)
    
    if kindmodele=="Mean":
        img = imageio.imread("img.png")
        s = np.mean(img)
        img_seg=  threshoding(img,s)
        
        st.title("Result of " +kindmodele + "   technic")
        result_show_c(img,img_seg,kindmodele,int(s))
    
    if kindmodele=="Median":
        img = imageio.imread("img.png")
        s = np.median(img)
        img_seg=  threshoding(img,s)

        st.title("Result of " +kindmodele + "   technic")
        result_show_c(img,img_seg,kindmodele,s)



#***********************contour*************************************************
elif choose=="Contour":
    contour_modele = ["Filtre de Sobel","Filtre de Prewitt","Filtre de Canny", "Filtre de Roberts", "Filtre de Scharr"]
    kindmodele = ""
    with st.sidebar:
        st.markdown("## **1. Import Image :open_file_folder:** ##") 
        datafile = st.file_uploader("Upload Image",type=None)
        st.markdown("## **2. Image Overwiew:mag:** ##") 
        if datafile is not None:
            st.write(type(datafile))
            file_details = {"filename":datafile.name,"filetype":datafile.type,"filesize":datafile.size}
            st.write(file_details)
            img = Image.open(datafile)
            img.save("img.png","png")

            # data = (datafile)
        
        st.markdown("## **3. Choose technic :computer:** ##") 
        kindmodele = st.sidebar.selectbox("What technic?", contour_modele)
    if kindmodele=="Filtre de Sobel":
        
       
        img = imageio.imread("img.png")
        sobel_img = sobel(img)
        st.title("Result of " +kindmodele + "   technic")
        result_show_r(img,sobel_img,kindmodele)
    
    if kindmodele=="Filtre de Prewitt":
        img = imageio.imread("img.png")
        prewitt_img = prewitt(img)
        st.title("Result of " +kindmodele + "   technic")
        result_show_r(img,prewitt_img,kindmodele)
    
    if kindmodele=="Filtre de Canny":
        img = imageio.imread("img.png")
        canny_img = cv2.Laplacian(img,cv2.CV_64F)
        st.title("Result of " +kindmodele + "   technic")
        result_show_r(img,canny_img,kindmodele)

    if kindmodele=="Filtre de Roberts":
        img = imageio.imread("img.png")
        roberts_img = roberts(img)
        st.title("Result of " +kindmodele + "   technic")
        result_show_r(img,roberts_img,kindmodele)
        
        
    if kindmodele=="Filtre de Scharr":
        img = imageio.imread("img.png")
        scharr_img = scharr(img)
        st.title("Result of " +kindmodele + "   technic")
        result_show_r(img,scharr_img,kindmodele)

   




# **************************region***************************************

elif choose=="Region": 

    region_modele = ["Kmeans","Fuzzy Cmeans","Filtre Gaussien", "Random Forest", "SVM" ]
    
    kindmodele = ""
    with st.sidebar:
        st.markdown("## **1. Import Image :open_file_folder:** ##") 
        datafile = st.file_uploader("Upload Image",type=None)
        st.markdown("## **2. Image Overwiew:mag:** ##") 
        if datafile is not None:
            st.write(type(datafile))
            # st.write(datafile[0])
            file_details = {"filename":datafile.name,"filetype":datafile.type,"filesize":datafile.size}
            st.write(file_details)
            img = Image.open(datafile)
            img.save("img.png","png")

            data = (datafile)
        
        st.markdown("## **3. Choose technic :computer:** ##") 
        kindmodele = st.sidebar.selectbox("What technic?", region_modele)
    
    if kindmodele=="Kmeans":
        classe = st.number_input("Numbers of classe", min_value=2, max_value=10, value=2)
        img = imageio.imread("img.png")
        img = img/255
        x = img.reshape(-1,1)
        knn = KMeans(n_clusters=int(classe))
        knn.fit(x)
        img_seg = knn.cluster_centers_
        img_seg = img_seg[knn.labels_]
        img_seg = img_seg.reshape(img.shape)
        # img_segm = imageio.imread(img_seg)
        # img_segm = Image.open(img_segm)
        # img_segm.save("img_seg.png","png")
        st.title("Result of " +kindmodele + "   technic")
        result_show_r(img,img_seg,kindmodele)
        
    if kindmodele=="Fuzzy Cmeans":
        classe = st.number_input("Numbers of classe", min_value=2, max_value=10, value=2)
        img = imageio.imread("img.png")
        x = img.reshape(-1,1)
        my_model = FCM(n_clusters=int(classe)) # we use two cluster as an example
        my_model.fit(x) 
        img_seg = my_model.centers
        labels = my_model.predict(x)
        img_seg = img_seg[labels]
        img_seg = img_seg.reshape(img.shape)
        st.title("Result of " +kindmodele + "   technic")
        result_show_r(img,img_seg,kindmodele)

    if kindmodele=="Filtre Gaussien":
        sigma = st.number_input("values of sigma", min_value=1, max_value=7, value=1)
        img = imageio.imread("img.png")
        gaussian_img = nd.gaussian_filter(img, sigma)
        st.title("Result of " +kindmodele + "   technic")
        result_show_r(img,gaussian_img,kindmodele)

    # if kindmodele=="Random Forest":
    #     filename = "random_forest_model"
    #     loaded_model = pickle.load(open(filename, 'rb'))
    #     img1= cv2.imread("img.png")
    #     img = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

    #     #Call the feature extraction function.
    #     X = feature_extraction(img)
    #     result = loaded_model.predict(X)
    #     segmented = result.reshape((img.shape))
    #     st.title("Result of " +kindmodele + "   technic")
    #     result_show_r(img,segmented,kindmodele)
    
    if kindmodele=="SVM":
        filename = "random_forest_model"
        loaded_model = pickle.load(open(filename, 'rb'))
        img1= cv2.imread("img.png")
        img = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

        #Call the feature extraction function.
        X = feature_extraction(img)
        result = loaded_model.predict(X)
        segmented = result.reshape((img.shape))
        st.title("Result of " +kindmodele + "   technic")
        result_show_r(img,segmented,kindmodele)
elif choose=="Home":
    st.markdown('<p class="first_titre"> Image Segmentation Platform</p>', unsafe_allow_html=True)
    st.write("---")
    
    
    st.write("##")
    st.markdown(
        '<p class="intro"><b>Here is The Image Segmentation Platform !</b></p>',
        unsafe_allow_html=True)

    c1, c2 = st.columns((3, 2))
    with c1:
            st.subheader("Team")
            st.write(
                "• AYITE KOMLAN")
            st.write(
                "• BAGUIAN HAROUNA")
    with c2:

        lottie_accueil = load_lottiefile('67523-teamwork-lottie-animation.json')
        st_lottie(lottie_accueil, height=200)