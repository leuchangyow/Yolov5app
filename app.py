from importlib.resources import path
import streamlit as st
from PIL import Image
import os
import shutil
from detect import main
import argparse
import sys
from pathlib import Path
import datetime

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


det=os.path.join(str(ROOT),'detect/')
tmp=os.path.join(str(ROOT),'tmp/')
if not os.path.exists(tmp): os.mkdir(tmp)
if not os.path.exists(det): os.mkdir(det)

if __name__ == '__main__':
    @st.cache
    def parse_opt(file_path,des_path,b):
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'ROI_detection.pt', help='model path(s)')
        parser.add_argument('--source', type=str, default=ROOT / file_path, help='file/dir/URL/glob, 0 for webcam')
        parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
        parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
        parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.3, help='NMS IoU threshold')
        parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true', help='show results')
        parser.add_argument('--save-txt', action='store_false', help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        parser.add_argument('--save-crop', action='store_false', help='save cropped prediction boxes')
        parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--visualize', action='store_true', help='visualize features')
        parser.add_argument('--update', action='store_true', help='update all models')
        parser.add_argument('--project', default=ROOT / des_path, help='save results to project/name')
        parser.add_argument('--name', default='exp', help='save results to project/name')
        parser.add_argument('--exist-ok', action=b, help='existing project/name ok, do not increment')
        parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
        parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
        parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
        parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
        parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
        opt = parser.parse_args()
        opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
        return opt


    
    st.set_page_config(
        page_title="Yolov5 Roi and Defect area detection",
        page_icon="random",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title('Yolov5 Roi and Defect area detection')
    
    source = ('Single image', 'All images in a folder')
    source_index = st.sidebar.selectbox(
        label = 'Select input', 
        options = range(len(source)),
        format_func = lambda x : source[x],
        help = 'select single image or a folder'
    )

    if source_index == 0:
        upload_file = st.sidebar.file_uploader(
            label='Select a image',
            type=['png','jpg'],
        )
        if upload_file is not None:
            is_valid= True
            is_folder=False
            with st.spinner('File uploading'):
                st.sidebar.image(upload_file)
                picture = Image.open(upload_file)
                if not os.path.exists(tmp+upload_file.name):
                    picture.save(tmp+upload_file.name)
                
        else:
            is_valid= False

    else:
        if len(os.listdir(det)) !=0:

            st.sidebar.write(f'There are {len(os.listdir(det)) } test folders under detect directory')
            folder=st.sidebar.selectbox(
                label='Select which folder you want to detect',
                options =os.listdir(det),
                help='There is a detect directory under Yolov5app, put your folder underneath'
            )            

            folder_path=os.path.join(det,folder) 
            is_valid=True
            is_folder=True
        else:
            st.sidebar.write('There are no folder under detect directory, please place image folder')
            is_valid=False
    
    if is_valid:
        if is_folder:
            ti=datetime.date.today()
            fp = os.path.join(str(ROOT),folder_path)
            dp = os.path.join(str(ROOT) ,'outcome/batch',str(ti))
            p=Path(dp)
            if not p.exists:
                p.mkdir()
            if st.button('Start detection:'):

                try:
                    main(parse_opt(fp,dp,'store_true'))
                    st.success('Detection complete')
                    st.text(f'The results of your inference will be located at {dp}')
                    st.text('Go and check it out!')
                    st.snow()
                except AssertionError:
                    st.error('No images found in this folder, please check and make sure the supporting format (jpg、png) is correct')
            else:
                pass
            
        else:
            ti=datetime.date.today()
            dp = os.path.join(str(ROOT) ,'outcome/single',str(ti))
            p=Path(dp)
            if not p.exists:
                p.mkdir()
            if st.button('Start detection'):
                main(parse_opt(tmp+upload_file.name,dp,'store_false'))
                st.success('Detection complete')
                st.text(f'The results of your inference will be located at {dp}')
                imag_show=Image.open(dp+'/exp/'+upload_file.name)
                st.image(imag_show)
                          
                st.snow()
            else:
                pass
        shutil.rmtree(tmp)
    else:
        pass



            


  
        
                
