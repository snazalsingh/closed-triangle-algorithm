#python detect_and_count.py --weights yolov7.pt --conf 0.1 --class 0 --source Bout-17-Sep-2022_10-59-48.avi
import argparse
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import os
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
bs = cv2.createBackgroundSubtractorMOG2()

person_count = ""

def count(founded_classes,im0):
  model_values=[]
  aligns=im0.shape
  align_bottom=aligns[0]
  align_right=(aligns[1]/1.7 ) 

  for i, (k, v) in enumerate(founded_classes.items()):
    a=f"{k} = {v}"
    model_values.append(v)
    align_bottom=align_bottom-35                                                   
    #cv2.putText(im0, str(a) ,(int(align_right),align_bottom), cv2.FONT_HERSHEY_SIMPLEX, 1,(45,255,255),1,cv2.LINE_AA)
  



def detect(save_img=True):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))
    
   

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                
                founded_classes={} # Creating a dict to storage our detected items
                # Print results
                for c in det[:, -1].unique():                 
                    n = (det[:, -1] == c).sum()  # detections per class                
                    class_index=int(c)
                    count_of_object=int(n)
                    
                    founded_classes[names[class_index]]=int(n)
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    count(founded_classes=founded_classes,im0=im0)  # Applying counter function
                 # Initialize a list to store the coordinates for each frame
              
           
              
                    

                counter=0 
                centroids=[]
                centroids_len=[]
                #def abc(counttt):
                    #cv2.putText(im0, f'People count: {counter}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                 # return
                def is_under_area(coord,AREA_COORDS):
                     return cv2.pointPolygonTest(np.array(AREA_COORDS), coord, False) >= 0
                def is_polygon_closed(pts):
                    return len(pts) >= 4 and np.array_equal(pts[0], pts[-1])  
                def save_image(im, counter):
                    filename = f'polygon_{counter}.jpg' 
                    cv2.imwrite(filename, im)
                    print(f'Saved image {filename} to {os.getcwd()}')
                # Write results
                
                for *xyxy, conf, cls in reversed(det):
                   if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                   #area_coords = [(0,0), (900,0), (900,900), (0,900)]
                   
                    #def discard_frame(frame):             
                  
                   #AREA_COORDS = [(91, 99), (948, 99), (959, 949), (111, 938)]
                   
                  # out = cv2.VideoWriter('output_video.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (im0.shape[1], im0.shape[0]))
                  # is_closed = False
                
                   MIN_CONF_LEVEL = 0.38
                   AREA_COORDS = [(91, 99), (930, 105), (944, 931), (111, 938)]
                  # box_id = 0
                # Iterate over each detected object (person)
                   for *xyxy, conf, cls in reversed(det):
                       
                       #cv2.putText(im0, "person" ,(300,600), cv2.FONT_HERSHEY_SIMPLEX, 1,(45,255,255),1,cv2.LINE_AA)
    # Extract the coordinates of the detected person
                       x1, y1, x2, y2 = map(int, xyxy)
                       x_center, y_center = (x1 + x2) // 2, (y1 + y2) // 2    
                       centroid = (x_center, y_center)
                       if is_under_area(centroid,AREA_COORDS) and conf >= MIN_CONF_LEVEL:
                            
                            #counter += 1
                           
                            
                            #count(founded_classes=founded_classes,im0=im0)
                           
                            #cv2.putText(im0, f'People count: {len(centroids)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                            centroids.append(centroid)
                            #box_id += 1
                            label = '{} {:.2f}'.format(names[int(cls)], conf)

                            area = (x2 - x1) * (y2 - y1) 
                            #box_id += 1
                            label += f'{area:.2f} pixels'
                            plot_one_box((x1, y1, x2, y2), im0, label=label, color=colors[int(cls)], line_thickness=1)

                   cv2.putText(im0, f'People count: {len(centroids)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                            
                            #abc(centroids_len.append(1))

                   
                    #if area > 25000 and counter > 1:
                                 #counter += 1
                            

                         
                         
                for i, c1 in enumerate(centroids):
                    for j, c2 in enumerate(centroids):
                        if i != j: 
                            dist = np.linalg.norm(np.array(c1) - np.array(c2))
                            print("distance",dist)     
                                
                            cv2.line(im0, c1, c2, (0, 0, 255), 2)
                            print(f"Distance between centroid {i} and {j}: {dist:.2f}")
                            #if len(centroids) == 3:
                                #for k, c3 in enumerate(centroids):
                                       #if i != k and j != k:
                                          #cv2.line(im0, c1, c3, (0, 0, 255), 2)
                                          #cv2.line(im0, c2, c3, (0, 0, 255), 2)
                                          
                      # for i, c in enumerate(centroids[:-1]):
                               # dist = np.linalg.norm(np.array(c) - np.array(centroid)) # calculate the distance between the centroids
                            
                                #cv2.line(im0, c, centroid, (0, 0, 255), 2)
                                #centroids_len[i] += 1
                               # centroids_len[-1] += 1
                        
                   #cv2.putText(im0, f'People count: {len(centroids)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  
                   #cv2.putText(img,str(len(totalCountUp)),(929,345),cv2.FONT_HERSHEY_PLAIN,5,(139,195,75),7)
                   #cv2.putText(img,str(len(totalCountDown)),(1191,345),cv2.FONT_HERSHEY_PLAIN,5,(50,50,230),7)
                   #pts = np.array(AREA_COORDS, np.int32)
                   #print("points ",pts)
                   #pts = pts.reshape((-1, 1, 2))
                   #print("pointssss",pts)
                   #cv2.polylines(im0, [pts], True, (0, 255, 255), 2) 

                   
                   
                if view_img:
                    cv2.imshow("Image", im0)
                    cv2.waitKey(1)
                    
                   #is_closed = False    

      
                       
                   
                           
                      
                           
                       
       
                           
                   
                       
                      
                                
                
                  
                   #if save_img:
                       #cv2.imwrite(save_path, im0)

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
            
            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                #save_path="E:\prg\Yolov7-Object-Counting"
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 70, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
