#############################################
# Object detection - YOLO - OpenCV
# Author : Arun Ponnusamy   (July 16, 2018)
# Website : http://www.arunponnusamy.com
############################################


import cv2
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image',
                help='path to input image')
ap.add_argument('-v', '--video',
                help='path to input video')
ap.add_argument('-c', '--config', required=True,
                help='path to yolo config file')
ap.add_argument('-w', '--weights', required=True,
                help='path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required=True,
                help='path to text file containing class names')
args = ap.parse_args()

detect_image = False
team_str = ["TEAM1", "TEAM2", "REFEREE"]


def get_output_layers(net):
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, team, x, y, x_plus_w, y_plus_h):
    # print(label)
    if team is not None:
        label=team
        idx = [i for i, name in enumerate(team_str) if name==label]
        color = COLORS[len(classes) + idx[0]]
    else:
        label = str(classes[class_id])
        color = COLORS[class_id]

    cv2.rectangle(img, (int(x), int(y)), (int(x_plus_w), int(y_plus_h)), color, 2)
    # print(x, y, x_plus_w, y_plus_h)

    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


print(args.video)
# cap = cv2.VideoCapture()
if (args.image):
    image = cv2.imread(args.image)

    Width = image.shape[1]
    Height = image.shape[0]
    detect_image = True
else:
    cap = cv2.VideoCapture(args.video)
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    fps_out = int(fps_in/2)
    vid_writer = cv2.VideoWriter("detected.mp4", cv2.VideoWriter_fourcc(*'MP4V'), fps_out,
                                 (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

scale = 0.00392

classes = None

with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes)+3, 3))

net = cv2.dnn.readNet(args.weights, args.config)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
# net = cv2.dnn.readNetFromDarknet(args.weights, args.config)
frameNum = 0


def get_player_info(rect, image, num):
    rect = [0 if r < 0 else r for r in rect]
    player = image[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
    h,w, ch = player.shape
    cx = rect[0] + rect[2]/2
    cy = rect[1] + rect[3]/2

    player = image[cy-h/3:cy+h/3, cx-w/3:cx+w/3]

    team1_range = np.array([[0,0,50], [50,60,255]])
    team2_range = np.array([[60, 0, 0],[255, 100, 70]])
    referee_range = np.array([[0,0,0], [30,30,30]])

    team1_mask = cv2.inRange(player, team1_range[0], team1_range[1])
    team2_mask = cv2.inRange(player, team2_range[0], team2_range[1])
    referee_mask = cv2.inRange(player, referee_range[0], referee_range[1])
    all_mask = [team1_mask, team2_mask, referee_mask]

    val_pert = [cv2.countNonZero(mask) for mask in all_mask]
    val_pert = [val*100/(player.shape[0] * player.shape[1]) for val in val_pert]
    if val_pert[2] > 15:
        cv2.imwrite("referee_{}.png".format(num), player)
        cv2.imwrite("referee_mask_{}.png".format(num), referee_mask)

    found = [team_str[i] for i,val in enumerate(val_pert) if val>15]

    print(val_pert)
    print("Player info {}".format(found))
    return (found[0] if found else None)

num=0
while (cv2.waitKey(1) < 0):

    if (detect_image and frameNum > 0):
        # cv2.waitKey(3000)
        break
    elif not detect_image:
        hasFrame, image = cap.read()
        Width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        Height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if not hasFrame:
            cv2.waitKey(3000)
            cap.release()
            break
    frameNum = frameNum + 1
    if(frameNum < 11*30):
        continue
    #if frameNum%3!=0:
    #    continue
    blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    team = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    print("--------------------------------------------------------")

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.65:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2

                if classes[class_id] == 'person':
                    num = num+1
                    teamname = get_player_info([x,y,w,h], image, num)
                else:
                    teamname = None

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
                team.append(teamname)
                print("{}:{}".format(classes[class_id], confidence))

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_prediction(image, class_ids[i], confidences[i], team[i], int(x), int(y), int(x + w), int(y + h))
    # t, _ = net.getPerfProfile()
    # label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
    # print(label)

    #cv2.imshow("object detection", image)
    if (args.image):
        cv2.imwrite("object-detection.jpg", image)
    else:
        vid_writer.write(image.astype(np.uint8))
cv2.destroyAllWindows()

print('Total num frames ', frameNum)
